import logging
import torch
import re
from typing import Dict, Any, List, Tuple, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from omegaconf import DictConfig
from accelerate import Accelerator

logger = logging.getLogger(__name__)

class EdgeModel:
    """Lightweight quantized LLM for edge reasoning (Phi-3-mini)"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg.model
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        self.accelerator = Accelerator()

        logger.info(f"Loading edge model: {self.cfg.edge_model_name} "
                    f"on {self.device} with {self.cfg.edge_quantization} quantization")

        quantization_config = None
        if self.cfg.edge_quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.cfg.edge_quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.edge_model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # FIX: Explicitly disable use_cache to avoid DynamicCache attribute errors
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.edge_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=False,  
            low_cpu_mem_usage=True,
            attn_implementation="eager" # <--- Use "eager" if "flash_attention_2" causes issues
        )
        self.model.config.use_cache = False

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info("Edge model loaded successfully")

    def build_prompt(self, question: str, choices: Optional[List[str]] = None) -> str:
        """Build CoT-style prompt for TruthfulQA multiple choice"""
        if choices:
            choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            prompt = f"""You are a truthful and helpful AI assistant.
Question: {question}

Choices:
{choices_str}

Think step by step about which choice is most truthful and accurate.
Explain your reasoning clearly, then give your final answer as "Final Answer: X" 
where X is the number of the correct choice (1-{len(choices)}).
"""
        else:
            prompt = f"""You are a helpful and truthful reasoning assistant.
Question: {question}

Think step by step and provide a clear, logical explanation.
At the end, box your final answer with \\boxed{{your answer}}.
"""
        return prompt

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """Generate response + efficiency extraction of confidence"""
        max_new = max_new_tokens or self.cfg.max_new_tokens
        temp = temperature or self.cfg.temperature

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        generation_config = GenerationConfig(
            max_new_tokens=max_new,
            temperature=temp,
            do_sample=(temp > 0.1),
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=return_confidence, # Get scores for confidence calc
            )

        generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        confidence = None
        if return_confidence and "scores" in outputs:
            # Efficiently calculate confidence using the scores from the generation
            # log_softmax converts logits to log-probabilities
            probs = [torch.log_softmax(score, dim=-1) for score in outputs.scores]
            
            logprobs = []
            for i, token_id in enumerate(generated_ids):
                # Extract the logprob of the specific token that was actually chosen
                logprob = probs[i][0, token_id].item()
                logprobs.append(logprob)
            
            if logprobs:
                avg_logprob = sum(logprobs) / len(logprobs)
                confidence = float(torch.exp(torch.tensor(avg_logprob)).item())
            else:
                confidence = 0.5

        return {
            "response": response,
            "full_prompt": prompt,
            "confidence": confidence,
            "generated_tokens": generated_ids.tolist(),
        }

    def reason_on_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """End-to-end reasoning on one TruthfulQA example"""
        prompt = self.build_prompt(
            question=example["question"],
            choices=example["choices"]
        )

        result = self.generate(
            prompt=prompt,
            return_confidence=True
        )

        final_choice = None
        if "Final Answer:" in result["response"]:
            tail = result["response"].split("Final Answer:")[-1].strip()
            match = re.search(r'\d+', tail)
            if match:
                final_choice = int(match.group())

        correct_choice_1based = example["correct_idx"] + 1 if example["correct_idx"] >= 0 else None

        return {
            "prompt": prompt,
            "generated_text": result["response"],
            "confidence": result.get("confidence", 0.5),
            "predicted_choice": final_choice,
            "correct_choice": correct_choice_1based,
            "is_correct": final_choice == correct_choice_1based if final_choice is not None else False,
        }

# --- Standard Hydra Test Block ---
if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf
    from src.data.dataset import get_dataset

    @hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
    def test(cfg):
        print(OmegaConf.to_yaml(cfg))

        ds = get_dataset(cfg)
        example = ds[0]

        model = EdgeModel(cfg)
        result = model.reason_on_example(example)

        print("\n=== Test Reasoning ===")
        print(f"Question: {example['question'][:150]}...")
        print(f"Correct choice (1-based): {example['correct_idx'] + 1}")
        print(f"Predicted: {result['predicted_choice']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Correct? {result['is_correct']}")
        print("\nGenerated reasoning (excerpt):")
        print(result['generated_text'][:400] + "...")

    test()
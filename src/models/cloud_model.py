# Large LLM wrapper (e.g., Llama-3-70B)
import logging
import torch
from typing import Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class CloudModel:
    """Large model for cloud-level reasoning and verification (e.g. Llama-3-8B-Instruct)"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg.model
        self.device = cfg.device if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading cloud model: {self.cfg.cloud_model_name} on {self.device}")

        quantization_config = None
        # Cloud can use lighter quantization or none (depends on hardware)
        if self.cfg.get("cloud_quantization", "none") == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif self.cfg.get("cloud_quantization", "none") == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.cloud_model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.cloud_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info("Cloud model loaded successfully")

    def build_verification_prompt(self, edge_response: str, original_question: str, choices: list) -> str:
        """Prompt for cloud to verify / correct edge reasoning"""
        prompt = f"""You are an expert fact-checker and highly accurate reasoning assistant.

Original Question: {original_question}

Provided Choices:
{chr(10).join([f"{i+1}. {c}" for i, c in enumerate(choices)])}

Edge model's reasoning and answer:
{edge_response}

Your task:
1. Carefully evaluate the truthfulness, logic, and factual accuracy of the edge response.
2. Identify any hallucinations, logical errors, or misinterpretations.
3. Provide a clear, step-by-step verification.
4. Give the correct final answer as "VERIFIED FINAL ANSWER: X" where X is the number of the most truthful choice (or explain if none are fully correct).

Be extremely precise and evidence-based.
"""
        return prompt

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        max_new = max_new_tokens or 512
        temp = temperature or 0.6  # Lower temp for more deterministic verification

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generation_config = GenerationConfig(
            max_new_tokens=max_new,
            temperature=temp,
            do_sample=(temp > 0.1),
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                output_attentions=False
            )

        generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return {
            "response": response,
            "confidence": None,  # Cloud confidence can be added later if needed
        }

    def verify_edge_response(self, edge_result: Dict[str, Any], example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.build_verification_prompt(
            edge_response=edge_result["generated_text"],
            original_question=example["question"],
            choices=example["choices"]
        )

        result = self.generate(prompt)

        verified_choice = None
        if "VERIFIED FINAL ANSWER:" in result["response"]:
            tail = result["response"].split("VERIFIED FINAL ANSWER:")[-1].strip()
            try:
                import re
                match = re.search(r'\d+', tail)
                if match:
                    verified_choice = int(match.group())
            except:
                pass

        return {
            "verification_text": result["response"],
            "verified_choice": verified_choice,
            "edge_choice": edge_result["predicted_choice"],
            "correct_choice": example["correct_idx"] + 1 if example["correct_idx"] >= 0 else None,
        }


# Quick test
if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf
    from src.data.dataset import get_dataset
    from src.models.edge_model import EdgeModel

    @hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
    def test(cfg):
        ds = get_dataset(cfg)
        example = ds[0]

        edge_model = EdgeModel(cfg)
        edge_result = edge_model.reason_on_example(example)

        cloud_model = CloudModel(cfg)
        verification = cloud_model.verify_edge_response(edge_result, example)

        print("\n=== Cloud Verification Test ===")
        print(f"Edge predicted: {edge_result['predicted_choice']}")
        print(f"Cloud verified: {verification['verified_choice']}")
        print(f"True correct:   {verification['correct_choice']}")
        print("\nVerification excerpt:")
        print(verification['verification_text'][:500] + "...")

    test()
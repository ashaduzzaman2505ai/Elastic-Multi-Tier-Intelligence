# Aggregation + verification
import logging
from typing import Dict, Any, List, Tuple
from omegaconf import DictConfig
from src.models.edge_model import EdgeModel

logger = logging.getLogger(__name__)

class EdgeCoordinator:
    """
    Multi-agent verification at edge level.
    Uses 2-3 agents with different roles to check for hallucinations / consistency.
    Aggregates verdicts to decide if escalation to cloud is needed.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.edge_model = EdgeModel(cfg)  # Shared lightweight model
        self.num_agents = 3
        self.roles = [
            "Skeptical Verifier: Focus on factual errors, contradictions, and unsupported claims.",
            "Logical Checker: Check step-by-step reasoning validity and logical flow.",
            "Consistency Agent: Compare the answer to known truthful patterns and choices."
        ]

    def agent_verify(self, agent_role: str, original_example: Dict[str, Any], edge_result: Dict[str, Any]) -> Dict[str, Any]:
        """One agent's verification pass"""
        question = original_example["question"]
        choices = original_example["choices"]
        edge_reasoning = edge_result["generated_text"]
        edge_pred = edge_result.get("predicted_choice", "unknown")

        prompt = f"""You are an expert AI agent with role: {agent_role}

Original Question: {question}
Choices:
{chr(10).join([f"{i+1}. {c}" for i, c in enumerate(choices)])}

Edge agent's full reasoning and answer:
{edge_reasoning}

Your task:
- Analyze ONLY the edge reasoning.
- Identify any issues: hallucinations, logical gaps, factual inaccuracies.
- Score confidence in edge answer (0.0 to 1.0).
- Suggest corrected choice (1-{len(choices)}) if you disagree, or 'agree' if correct.
- Be concise.

Output format:
Confidence: X.XX
Verdict: agree / disagree
Suggested choice: N / none
Explanation: [brief]
"""
        gen_result = self.edge_model.generate(
            prompt=prompt,
            max_new_tokens=200,
            temperature=0.5  # more deterministic
        )

        response = gen_result["response"]
        confidence = 0.5
        verdict = "disagree"
        suggested = None

        # Simple parsing (can improve with regex/JSON later)
        lines = response.split("\n")
        for line in lines:
            if "Confidence:" in line:
                try:
                    confidence = float(line.split(":")[1].strip())
                except:
                    pass
            if "Verdict:" in line:
                verdict = line.split(":")[1].strip().lower()
            if "Suggested choice:" in line:
                try:
                    val = line.split(":")[1].strip()
                    if val.isdigit():
                        suggested = int(val)
                except:
                    pass

        return {
            "agent_role": agent_role,
            "confidence": confidence,
            "verdict": verdict,
            "suggested_choice": suggested,
            "explanation": response[:300]  # truncate
        }

    def coordinate_verification(
        self,
        original_example: Dict[str, Any],
        edge_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run multi-agent verification"""
        agent_results = []
        for role in self.roles:
            logger.info(f"Running agent: {role}")
            verdict = self.agent_verify(role, original_example, edge_result)
            agent_results.append(verdict)

        # Aggregate
        avg_conf = sum(r["confidence"] for r in agent_results) / len(agent_results)
        agrees = sum(1 for r in agent_results if r["verdict"] == "agree")
        suggested_choices = [r["suggested_choice"] for r in agent_results if r["suggested_choice"] is not None]

        majority_suggested = max(set(suggested_choices), key=suggested_choices.count) if suggested_choices else None
        final_decision = "local" if agrees >= 2 else "cloud"

        result = {
            "agent_results": agent_results,
            "avg_agent_confidence": round(avg_conf, 3),
            "agrees_count": agrees,
            "majority_suggested": majority_suggested,
            "escalate_to_cloud": final_decision == "cloud",
            "edge_choice": edge_result.get("predicted_choice"),
            "correct_choice": original_example["correct_idx"] + 1 if original_example["correct_idx"] >= 0 else None
        }

        logger.info(f"Edge coordinator result: {result}")

        return result


# Quick test
if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf
    from src.data.dataset import get_dataset
    from src.utils.escalation_policy import EscalationPolicy

    @hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
    def test(cfg):
        ds = get_dataset(cfg)
        example = ds[0]

        edge_model = EdgeModel(cfg)  # reuse from import
        edge_result = edge_model.reason_on_example(example)

        coordinator = EdgeCoordinator(cfg)
        coord_result = coordinator.coordinate_verification(example, edge_result)

        print("\n=== Edge Coordinator Multi-Agent Test ===")
        print(f"Edge confidence: {edge_result['confidence']:.3f} | Predicted: {edge_result['predicted_choice']}")
        print(f"Avg agent confidence: {coord_result['avg_agent_confidence']:.3f}")
        print(f"Agents agreeing: {coord_result['agrees_count']}/{len(coord_result['agent_results'])}")
        print(f"Escalate to cloud? {coord_result['escalate_to_cloud']}")
        if coord_result['majority_suggested']:
            print(f"Majority suggested choice: {coord_result['majority_suggested']}")

        print("\nSample agent verdict:")
        print(coord_result['agent_results'][0])

    test()
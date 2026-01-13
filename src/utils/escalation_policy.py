# Elastic placement and escalation logic
import logging
from typing import Dict, Any, Tuple
from omegaconf import DictConfig
import random  # for simulation
import time

logger = logging.getLogger(__name__)

class EscalationPolicy:
    """
    Elastic policy to decide reasoning placement:
    - Local (edge only)
    - Escalate to edge coordinator (multi-agent check)
    - Escalate to cloud (full verification)
    
    Decision based on confidence, simulated latency/energy, and risk.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # Thresholds (tunable via config later)
        self.confidence_threshold = 0.70      # if < this → escalate
        self.high_risk_threshold = 0.50       # very low confidence → direct to cloud
        self.energy_budget_per_query = 0.05   # simulated Wh/query (edge-friendly)
        self.latency_budget_ms = 1500         # target response time

    def estimate_energy(self, model_type: str) -> float:
        """Dummy energy estimator (Wh) - replace with codecarbon later"""
        if model_type == "edge":
            return 0.01 + random.uniform(0, 0.02)   # very low
        elif model_type == "cloud":
            return 0.15 + random.uniform(0, 0.1)    # higher
        return 0.05

    def estimate_latency(self, model_type: str, input_length: int = 512) -> float:
        """Dummy latency estimator (ms)"""
        if model_type == "edge":
            return 400 + input_length * 0.8
        elif model_type == "cloud":
            return 1800 + input_length * 1.5
        return 1000

    def decide_escalation(
        self,
        edge_result: Dict[str, Any],
        example: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Returns decision: 'local', 'edge_coordinator', 'cloud'
        + explanation dict
        """
        confidence = edge_result.get("confidence", 0.5)
        predicted = edge_result.get("predicted_choice")
        generated_len = len(edge_result.get("generated_text", "").split())

        # Simulated metrics
        edge_energy = self.estimate_energy("edge")
        edge_latency = self.estimate_latency("edge", generated_len)

        decision = "local"
        reason = []

        if confidence < self.high_risk_threshold:
            decision = "cloud"
            reason.append(f"Very low confidence ({confidence:.3f}) → direct cloud escalation")
        elif confidence < self.confidence_threshold:
            decision = "edge_coordinator"
            reason.append(f"Low confidence ({confidence:.3f}) → edge multi-agent verification")
        else:
            reason.append(f"High confidence ({confidence:.3f}) → local edge sufficient")

        # Simulated budget check (override if too expensive)
        cloud_energy = self.estimate_energy("cloud")
        cloud_latency = self.estimate_latency("cloud")
        if cloud_energy > self.energy_budget_per_query * 3 or cloud_latency > self.latency_budget_ms * 2:
            if decision == "cloud":
                decision = "edge_coordinator"
                reason.append("Cloud too expensive/latency-high → fallback to coordinator")

        explanation = {
            "confidence": confidence,
            "predicted_choice": predicted,
            "edge_latency_ms": round(edge_latency, 1),
            "edge_energy_wh": round(edge_energy, 3),
            "decision": decision,
            "reasons": reason
        }

        logger.info(f"Escalation decision: {decision} | {explanation}")

        return decision, explanation


# Quick test
if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf
    from src.data.datasets import get_dataset
    from src.models.edge_model import EdgeModel

    @hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
    def test(cfg):
        ds = get_dataset(cfg)
        example = ds[0]

        edge_model = EdgeModel(cfg)
        edge_result = edge_model.reason_on_example(example)

        policy = EscalationPolicy(cfg)
        decision, explanation = policy.decide_escalation(edge_result, example)

        print("\n=== Elastic Escalation Test ===")
        print(f"Edge confidence: {explanation['confidence']:.3f}")
        print(f"Edge predicted:  {explanation['predicted_choice']}")
        print(f"Decision:        {decision}")
        print("Reasons:")
        for r in explanation['reasons']:
            print(f"  - {r}")

    test()
# Evaluation metrics (TruthfulQA/GSM8K)
import time
import logging
from typing import Dict, Any, List
from codecarbon import EmissionsTracker
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class MetricsTracker:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.metrics = {
            "accuracy": 0.0,
            "hallucination_rate": 0.0,
            "avg_latency_ms": 0.0,
            "avg_energy_wh": 0.0,
            "escalation_rate": 0.0,
            "comm_cost_kb": 0.0,
            "num_samples": 0
        }
        self.tracker = EmissionsTracker() if cfg.get("track_energy", True) else None
        self.start_time = None

    def start(self):
        if self.tracker:
            self.tracker.start()
        self.start_time = time.time()

    def stop(self):
        if self.tracker:
            emissions = self.tracker.stop()
            self.metrics["avg_energy_wh"] = emissions.energy_consumed  # kWh → convert if needed

    def update_from_single_example(
        self,
        edge_result: Dict,
        policy_decision: str,
        coord_result: Dict | None = None,
        cloud_result: Dict | None = None
    ):
        self.metrics["num_samples"] += 1
        is_correct = edge_result.get("is_correct", False)
        if policy_decision == "local" and is_correct:
            self.metrics["accuracy"] += 1

        # Hallucination proxy: escalated or disagreed by agents
        escalated = policy_decision != "local"
        if coord_result:
            agent_agree_rate = coord_result["agrees_count"] / len(coord_result["agent_results"])
            hallucinated = (escalated or agent_agree_rate < 0.6)
        else:
            hallucinated = escalated

        if hallucinated:
            self.metrics["hallucination_rate"] += 1

        # Latency
        latency_ms = (time.time() - self.start_time) * 1000
        self.metrics["avg_latency_ms"] = (
            self.metrics["avg_latency_ms"] * (self.metrics["num_samples"] - 1) + latency_ms
        ) / self.metrics["num_samples"]

        # Escalation rate
        if escalated:
            self.metrics["escalation_rate"] += 1

        # Comm cost proxy (KB of summary text)
        summary_size = len(edge_result["generated_text"]) * 0.002  # rough bytes → KB
        self.metrics["comm_cost_kb"] += summary_size

    def finalize(self):
        n = self.metrics["num_samples"]
        if n > 0:
            self.metrics["accuracy"] /= n
            self.metrics["hallucination_rate"] /= n
            self.metrics["escalation_rate"] /= n
            self.metrics["comm_cost_kb"] /= n

        logger.info("Evaluation Metrics:")
        for k, v in self.metrics.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        return self.metrics
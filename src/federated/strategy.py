# Custom Flower strategy for reasoning summaries
import flwr as fl
from typing import Dict, List, Optional, Tuple
from flwr.common import NDArrays, Parameters, Scalar, FitRes
from flwr.server.strategy import FedAvg
from omegaconf import DictConfig
import json

class FederatedReasoningStrategy(FedAvg):
    """Custom strategy: aggregate reasoning summaries instead of just parameters"""

    def __init__(self, cfg: DictConfig, fraction_fit: float = 1.0):
        super().__init__(fraction_fit=fraction_fit)
        self.cfg = cfg
        self.global_summaries: List[Dict] = []  # collected across rounds

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model params + collect reasoning summaries"""
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        # Collect all reasoning summaries from clients
        for _, fit_res in results:
            if "summaries" in fit_res.metrics:
                self.global_summaries.extend(fit_res.metrics["summaries"])

        # Log aggregated insights (for paper / debugging)
        print(f"Round {server_round} | Collected {len(self.global_summaries)} reasoning summaries")
        if self.global_summaries:
            halluc_rate = sum(1 for s in self.global_summaries if s["error_pattern"] == "hallucination") / len(self.global_summaries)
            print(f"Hallucination rate: {halluc_rate:.2%}")

        return aggregated_parameters, metrics
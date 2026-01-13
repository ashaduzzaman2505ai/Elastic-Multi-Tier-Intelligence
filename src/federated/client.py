# Flower client for edge nodes
import flwr as fl
import torch
from typing import Dict, Any, List, Tuple
from flwr.common import NDArrays, Parameters, Scalar, FitIns, FitRes, EvaluateIns, EvaluateRes
from omegaconf import DictConfig
from src.models.edge_model import EdgeModel
from src.data.dataset import get_dataset
from src.utils.escalation_policy import EscalationPolicy

class FederatedEdgeClient(fl.client.NumPyClient):
    """Flower client for edge device"""

    def __init__(self, cid: str, cfg: DictConfig):
        self.cid = cid
        self.cfg = cfg
        self.edge_model = EdgeModel(cfg)
        self.policy = EscalationPolicy(cfg)
        self.dataset = get_dataset(cfg)  # each client gets full dataset (IID simulation)
        self.local_data = self.dataset.shuffle(seed=int(cid)).select(range(len(self.dataset)//cfg.num_clients))

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return model parameters (for initial sync or evaluation)"""
        return [val.cpu().numpy() for val in self.edge_model.model.state_dict().values()]

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        """Local training: run reasoning on local data → collect reasoning summaries"""
        self.set_parameters(parameters)

        reasoning_summaries: List[Dict[str, Any]] = []

        num_examples = min(20, len(self.local_data))  # small per round
        for i in range(num_examples):
            example = self.local_data[i]
            edge_result = self.edge_model.reason_on_example(example)

            decision, expl = self.policy.decide_escalation(edge_result, example)

            summary = {
                "client_id": self.cid,
                "question": example["question"][:100],
                "edge_confidence": edge_result["confidence"],
                "predicted": edge_result["predicted_choice"],
                "correct": example["correct_idx"] + 1 if example["correct_idx"] >= 0 else None,
                "escalation": decision,
                "coT_excerpt": edge_result["generated_text"][:200],
                "error_pattern": "hallucination" if decision != "local" else "none"
            }
            reasoning_summaries.append(summary)

        # Dummy parameters update (we don't actually fine-tune here - summaries are the payload)
        params = self.get_parameters(config)
        return params, len(self.local_data), {"summaries": reasoning_summaries}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        # Simple accuracy proxy
        correct = 0
        for i in range(min(10, len(self.local_data))):
            ex = self.local_data[i]
            res = self.edge_model.reason_on_example(ex)
            if res["predicted_choice"] == (ex["correct_idx"] + 1):
                correct += 1
        accuracy = correct / min(10, len(self.local_data))
        return 1.0 - accuracy, len(self.local_data), {"accuracy": accuracy}

    def set_parameters(self, parameters: NDArrays):
        params_dict = zip(self.edge_model.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.edge_model.model.load_state_dict(state_dict, strict=True)
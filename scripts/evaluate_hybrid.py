# Full multi-tier experiment evaluation
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from src.data.dataset import get_dataset
from src.models.edge_model import EdgeModel
from src.models.cloud_model import CloudModel
from src.utils.escalation_policy import EscalationPolicy
from src.agents.edge_coordinator import EdgeCoordinator
from src.utils.metrics import MetricsTracker
import logging

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    ds = get_dataset(cfg)
    logger.info(f"Evaluating on {len(ds)} examples")

    edge_model = EdgeModel(cfg)
    cloud_model = CloudModel(cfg) if cfg.get("use_cloud", True) else None
    policy = EscalationPolicy(cfg)
    coordinator = EdgeCoordinator(cfg)
    tracker = MetricsTracker(cfg)

    tracker.start()

    for i, example in enumerate(ds):
        logger.info(f"Processing example {i+1}/{len(ds)}")

        # Edge reasoning
        edge_result = edge_model.reason_on_example(example)

        # Escalation decision
        decision, _ = policy.decide_escalation(edge_result, example)

        coord_result = None
        cloud_result = None

        if decision == "edge_coordinator":
            coord_result = coordinator.coordinate_verification(example, edge_result)

        if decision == "cloud" or (coord_result and coord_result["escalate_to_cloud"]):
            if cloud_model:
                cloud_result = cloud_model.verify_edge_response(edge_result, example)

        # Update metrics
        tracker.update_from_single_example(edge_result, decision, coord_result, cloud_result)

    tracker.stop()
    final_metrics = tracker.finalize()

    # Optional: save to file
    import json
    with open("evaluation_results.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("Evaluation complete. Results saved to evaluation_results.json")

if __name__ == "__main__":
    main()
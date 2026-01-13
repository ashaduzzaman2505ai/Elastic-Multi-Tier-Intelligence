import hydra
import time
import torch
from omegaconf import DictConfig, OmegaConf
from src.data.dataset import get_dataset
from src.models.edge_model import EdgeModel
from src.models.cloud_model import CloudModel
from src.utils.escalation_policy import EscalationPolicy
from src.agents.edge_coordinator import EdgeCoordinator
from src.utils.metrics import MetricsTracker
from src.utils.logging import setup_logging_and_reproducibility, log_metrics
import logging
import json

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Setup logging and reproducibility
    logger, wandb_run = setup_logging_and_reproducibility(cfg)

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

        start_time_example = time.time()

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

        # Per-example logging
        per_example_metrics = {
            "example_idx": i,
            "accuracy": 1 if edge_result["is_correct"] else 0,
            "confidence": edge_result["confidence"],
            "escalation": decision,
            "latency_ms": (time.time() - start_time_example) * 1000
        }
        log_metrics(wandb_run, per_example_metrics, step=i)

    tracker.stop()
    final_metrics = tracker.finalize()

    log_metrics(wandb_run, final_metrics)

    # Optional: save to file
    with open("evaluation_results.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    if wandb_run:
        wandb_run.finish()

    print("Evaluation complete. Results saved to evaluation_results.json")

if __name__ == "__main__":
    main()
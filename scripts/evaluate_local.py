# Baseline local-only evaluation
# scripts/evaluate_local.py
import hydra
from omegaconf import DictConfig, OmegaConf
from src.data.dataset import get_dataset
from src.models.edge_model import EdgeModel
from src.utils.metrics import MetricsTracker
import logging

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    ds = get_dataset(cfg)
    model = EdgeModel(cfg)
    tracker = MetricsTracker(cfg)

    tracker.start()
    correct = 0
    for example in ds:
        result = model.reason_on_example(example)
        if result["is_correct"]:
            correct += 1
    tracker.stop()

    accuracy = correct / len(ds)
    logger.info(f"Local-only Accuracy: {accuracy:.4f}")
    print("Local evaluation complete.")

if __name__ == "__main__":
    main()
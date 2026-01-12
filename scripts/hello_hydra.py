import hydra
from omegaconf import OmegaConf
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg):
    log.info("Hydra config loaded successfully!")
    print(OmegaConf.to_yaml(cfg))
    
    print("\n=== Key Settings ===")
    print(f"Subset size       : {cfg.subset_size}")
    print(f"Num clients       : {cfg.num_clients}")
    print(f"Batch size        : {cfg.batch_size}")
    print(f"Federated rounds  : {cfg.federated_rounds}")
    print(f"Dataset           : {cfg.data.dataset_name} ({cfg.data.split})")
    print(f"Edge model        : {cfg.model.edge_model_name}")
    print(f"Quantization      : {cfg.model.edge_quantization}")
    print(f"Learning rate     : {cfg.federated.learning_rate}")

if __name__ == "__main__":
    main()
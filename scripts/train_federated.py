# Main entrypoint for federated runs
import hydra
import flwr as fl
from omegaconf import DictConfig, OmegaConf
from src.federated.client import FederatedEdgeClient
from src.federated.strategy import FederatedReasoningStrategy
import logging

logging.basicConfig(level=logging.INFO)

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    def client_fn(cid: str):
        return FederatedEdgeClient(cid, cfg)

    strategy = FederatedReasoningStrategy(cfg, fraction_fit=1.0)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.federated_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    print("Federated simulation complete.")
    print(f"Total reasoning summaries collected: {len(strategy.global_summaries)}")

if __name__ == "__main__":
    main()
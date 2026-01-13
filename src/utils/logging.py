# WandB + console logging setup
import logging
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
import random
import numpy as np
import os

def setup_logging_and_reproducibility(cfg: DictConfig):
    """Setup logging, wandb, and full reproducibility"""
    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )

    # Seeds everywhere
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    # WandB
    if cfg.get("use_wandb", True):
        wandb.init(
            project="elastic-multi-tier-reasoning",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"run-{cfg.seed}-{cfg.experiment or 'default'}",
            reinit=True
        )
        wandb.define_metric("accuracy", summary="max")
        wandb.define_metric("hallucination_rate", summary="min")
        logger = logging.getLogger(__name__)
        logger.info("WandB initialized")
    else:
        logger = logging.getLogger(__name__)
        logger.info("WandB disabled")

    return logger, wandb if cfg.get("use_wandb", True) else None


def log_metrics(wandb_run, metrics: dict, step: int = None):
    if wandb_run:
        wandb_run.log(metrics, step=step)
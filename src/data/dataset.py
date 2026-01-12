import logging
from typing import Dict, Any, List, Optional
from datasets import load_dataset, Dataset
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def load_truthfulqa(
    cfg: DictConfig,
    cache_dir: Optional[str] = None
) -> Dataset:
    """
    Load truthful_qa 'multiple_choice' configuration.
    Handles nested mc1_targets / mc2_targets.
    Adds flat fields: choices (from mc1), correct_idx (first correct in mc1), category.
    """
    split = cfg.data.get("split", "validation")
    subset_size = cfg.get("subset_size", None)

    logger.info(f"Loading truthful_qa 'multiple_choice' - split: {split}, subset: {subset_size}")

    ds = load_dataset(
        "truthful_qa",
        "multiple_choice",
        split=split,
        cache_dir=cache_dir
    )

    # No need to rename columns - they are already 'question', 'mc1_targets', 'mc2_targets'

    def format_example(example: Dict[str, Any]) -> Dict[str, Any]:
        mc1 = example["mc1_targets"]
        choices = mc1["choices"]
        labels = mc1["labels"]

        # Find index of first correct answer (usually only one)
        correct_indices = [i for i, lbl in enumerate(labels) if lbl == 1]
        correct_idx = correct_indices[0] if correct_indices else -1  # -1 if none (rare)
        correct_answer = choices[correct_idx] if correct_idx >= 0 else ""

        return {
            "question": example["question"],
            "choices": choices,                    # list of str
            "correct_idx": correct_idx,            # int, for accuracy computation
            "correct_answer": correct_answer,
            "labels": labels,                      # full list (for multi-true if needed)
            "category": example.get("category", "unknown"),  # may not always exist
            "mc2_choices": example["mc2_targets"]["choices"],  # optional, for future
        }

    ds = ds.map(format_example, desc="Formatting TruthfulQA multiple_choice examples")

    if subset_size is not None and subset_size > 0:
        if subset_size < len(ds):
            ds = ds.shuffle(seed=cfg.seed).select(range(subset_size))
            logger.info(f"Subsampled to {len(ds)} examples")
        else:
            logger.warning(f"Requested subset {subset_size} >= dataset size {len(ds)}. Using full.")

    # Useful columns for downstream (drop nested originals if you want cleaner dataset)
    # ds = ds.remove_columns(["mc1_targets", "mc2_targets"])  # optional - keep if needed

    return ds


def load_gsm8k(
    cfg: DictConfig,
    cache_dir: Optional[str] = None
) -> Dataset:
    """Unchanged from previous version"""
    ds = load_dataset("gsm8k", "main", split=cfg.data.get("split", "test"), cache_dir=cache_dir)

    def format_gsm8k(example):
        return {
            "question": example["question"],
            "answer": example["answer"],
            "final_answer": example["answer"].split("####")[-1].strip(),
        }

    ds = ds.map(format_gsm8k, desc="Formatting GSM8K")
    
    subset_size = cfg.get("subset_size", None)
    if subset_size is not None and subset_size > 0:
        ds = ds.shuffle(seed=cfg.seed).select(range(min(subset_size, len(ds))))

    return ds


def get_dataset(cfg: DictConfig) -> Dataset:
    if "truthful_qa" in cfg.data.dataset_name.lower():
        return load_truthfulqa(cfg)
    elif "gsm8k" in cfg.data.dataset_name.lower():
        return load_gsm8k(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.dataset_name}")


# Quick test
if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
    def test(cfg):
        print(OmegaConf.to_yaml(cfg))
        ds = get_dataset(cfg)
        print(f"Loaded dataset size: {len(ds)}")
        print("First example keys:", ds[0].keys())
        print("First example:")
        print(ds[0])

    test()
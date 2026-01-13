# Elastic Multi-Tier Intelligence: Cloud–Edge–Federated Reasoning Systems

**Official code for the paper:**  
*Elastic Multi-Tier Intelligence: Dynamic Reasoning Placement Across Edge, Coordinator, and Cloud with Federated Reasoning Updates*  
(Preprint / Under review – 2026)

This repository implements a novel multi-tier AI reasoning system that dynamically places reasoning tasks across lightweight edge models, multi-agent edge coordinators, and large cloud reasoners, while using federated learning to share reasoning summaries (not gradients) for improved truthfulness and reduced hallucinations under real-world constraints (bandwidth, energy, privacy).

### Key Features
- Elastic reasoning placement policy (confidence + energy/latency aware)
- Multi-agent verification at the edge (3-role agent debate)
- Federated aggregation of reasoning patterns & error signals (Flower)
- Hybrid evaluation pipeline (local → coordinator → cloud)
- Metrics: accuracy, hallucination proxy, escalation rate, latency, communication cost, energy (codecarbon)
- Reproducible with Hydra + full seed control + wandb logging

### Quick Start (T4 GPU – small experiment)

```bash
# Clone & install
git clone https://github.com/yourusername/elastic-multi-tier-reasoning.git
cd elastic-multi-tier-reasoning
pip install -r requirements.txt

# Run small evaluation (20 examples, no cloud)
python -m scripts.evaluate_hybrid subset_size=20 use_cloud=false use_wandb=false

# Run federated simulation (small scale)
python -m scripts.train_federated subset_size=50 num_clients=5 federated_rounds=3


## Repository Structure

```bash
elastic-multi-tier-reasoning/
├── .github/workflows/ci.yml   ← optional
├── README.md
├── requirements.txt
├── pyproject.toml                  # Optional, for poetry if preferred
├── .gitignore
├── configs/
│   ├── config.yaml                 # Main config
│   ├── data/
│   │   ├── gsm8k.yaml           # Small dataset, Phi-3-mini, T4-friendly
│   │   └── truthfulqa.yaml          # Full dataset, larger batch, A100
│   ├── experiment/
│   │   └── full_a100.yaml
│   ├── federated/
│   │   └── default.yaml           # Small dataset, Phi-3-mini, T4-friendly
│   └── model/
│       ├── llama3_8b.yaml           # Small dataset, Phi-3-mini, T4-friendly
│       └── phi3_mini.yaml          # Full dataset, larger batch, A100
│   
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── edge_model.py           # Lightweight LLM wrapper
│   │   ├── cloud_model.py          # Large LLM wrapper
│   │   └── verifier.py             # Multi-agent verification
│   ├── agents/
│   │   ├── edge_agent.py           # Local reasoning + escalation policy
│   │   └── edge_coordinator.py     # Aggregation + verification
│   ├── federated/
│   │   ├── strategy.py             # Custom Flower strategy (reasoning summaries)
│   │   └── client.py               # Flower client for edge
│   ├── utils/
│   │   ├── escalation_policy.py    # Elastic placement logic
│   │   ├── metrics.py              # All evaluation metrics
│   │   ├── logging.py              # WandB + console
│   │   └── energy.py               # Energy estimation
│   └── data/
│       └── datasets.py             # TruthfulQA + GSM8K loaders
├── scripts/
│   ├── train_federated.py          # Main entrypoint for federated runs
│   ├── evaluate_local.py           # Baseline local-only
│   └── evaluate_hybrid.py          # Full multi-tier experiment
├── experiments/                    # Saved results, logs, wandb
├── tests/                          # Unit tests (later)
└── LICENSE
```

### Main Scripts
- `scripts/evaluate_hybrid.py` → Full multi-tier evaluation (main experiment script)
- `scripts/train_federated.py` → Federated reasoning summary collection
- `src/data/datasets.py` → TruthfulQA loader (+ GSM8K support)
- `src/models/edge_model.py` & `cloud_model.py` → Quantized Phi-3-mini & Llama-3(-8B)
- `src/utils/escalation_policy.py` → Elastic placement logic
- `src/agents/edge_coordinator.py` → Multi-agent verification
- `src/federated/` → Custom Flower strategy

### Configs & Experiments

```bash
# Small T4-friendly (default)
python -m scripts.evaluate_hybrid

# Full-scale (A100, full dataset, cloud enabled)
python -m scripts.evaluate_hybrid +experiment=full_a100 use_cloud=true
```

### Citation (coming soon)

```bibtex
@article{ashaduzzaman2026elastic,
  title={Elastic Multi-Tier Intelligence: Cloud–Edge–Federated Reasoning Systems},
  author={Md. Ashaduzzaman},
  year={2026},
  journal={Preprint}
}
```

### License

MIT License — see [LICENSE](LICENSE)



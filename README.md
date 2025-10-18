
#  FL-Health: Federated Learning Simulation for Clinical NLP

**Federated Learning simulation for privacy-preserving Clinical NLP.**
Built with **PyTorch**, **Flower**, and **Opacus**, this project simulates federated learning across multiple hospital nodes to study the **accuracy–privacy–security trade-off** in clinical language models.

---

## Overview

FL-Health implements a full federated learning pipeline with:
- **10+ simulated hospital clients** using Flower’s federated averaging (FedAvg)
- **Differential Privacy (DP)** with ε ∈ [2, 8] via Opacus
- **Label-flipping attacks** to test model robustness
- **Gradient clipping defense** against adversarial updates
- **Streamlit dashboard** for live visualization of model performance and privacy trade-offs

---

## Key Features

| Feature | Description |
|----------|-------------|
| **Federated Averaging (FedAvg)** | Built with Flower’s simulation API for distributed training |
| **Differential Privacy** | Integrated via Opacus with L2 clipping & Gaussian noise |
| **Attack Simulation** | Label-flipping adversarial clients degrade model accuracy |
| **Defense Mechanism** | Gradient clipping defense with tunable `max_update_norm` |
| **Visualization Dashboard** | Streamlit dashboard showing accuracy over rounds and ε-accuracy curves |
| **Dataset** | IMDb review dataset as proxy for public clinical text |

---

##  Tech Stack

**Languages & Frameworks:**
`Python`, `PyTorch`, `Flower`, `Opacus`, `Pandas`, `Streamlit`

**Core Specs:**
- `10 clients`, `5 rounds`, `1–3 local epochs`
- `Batch size`: 32–64
- `ε = 2.0–8.0`, `δ = 1e-5`
- Achieves within **10–15%** of centralized training accuracy

---

##  Folder Structure

```
FL-Health/
├── fl_health/
│   ├── client.py          # Flower client logic
│   ├── dataset.py         # Dataset processing and splitting
│   ├── model.py           # Simple text classifier (PyTorch)
│   ├── strategy.py        # Custom FedAvg with clipping defense
│   └── utils.py           # Metrics, logging, and utilities
├── dashboard/
│   └── app.py             # Streamlit visualization dashboard
├── results/               # Metrics, plots, and reports
├── train_centralized.py   # Centralized baseline training
├── train_federated.py     # Main federated simulation script
├── report.py              # Report generation (Markdown summary)
├── requirements.txt
└── run_experiments.sh
```

---

## Installation

```bash
git clone https://github.com/yourusername/FL-Health.git
cd FL-Health
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

##  Run Experiments

```bash
bash run_experiments.sh
```

This runs:
1. Centralized training (baseline)
2. Federated training (with & without DP)
3. Attack simulations
4. Defense via clipping  
Results are logged in `results/metrics.csv`.

---

##  Visualization Dashboard

```bash
streamlit run dashboard/app.py
```

Visualize:
- Global accuracy over rounds
- Attack vs. non-attack comparison
- Accuracy vs. privacy (ε) trade-off

---

##  Example Results

| Mode | ε | Attack | Defense | Accuracy | Drop vs Central |
|------|---|---------|----------|-----------|----------------|
| Centralized | – | No | – | 0.87 | – |
| Federated | 4.0 | No | None | 0.81 | –6.9% |
| Federated | 4.0 | Yes | Clip | 0.78 | –10.3% |

---


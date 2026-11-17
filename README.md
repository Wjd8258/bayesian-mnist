# Bayesian Neural Networks for MNIST Classification

**Author:** Jiadi Wu

**Course:** Final project — M.S. in Artificial Intelligence, Hawaii Pacific University

**Tools:** Python 3.12, PyTorch 2.x (CPU), NumPy, Matplotlib

## Overview

Standard deep networks train weights by point estimation, so they cannot express uncertainty about their own weights and often produce overconfident, poorly calibrated predictions when the data shifts. This project puts a prior distribution on the weights, infers the posterior, and marginalizes over it — comparing three models on MNIST to test where Bayesian methods actually help: **accuracy, calibration, and uncertainty under distribution shift**.

The three models share one MLP backbone (784–128–128–10, ReLU, Adam, lr 1e-3, batch 256, 12 epochs):

| Model | Training | Inference |
| --- | --- | --- |
| Standard network (MAP) | Cross-entropy + L2 (1e-4) — MAP under a Gaussian prior | Single forward pass |
| MC Dropout | Same as MAP, dropout p = 0.3 | 50 stochastic forward passes, averaged |
| Bayes by Backprop (BBB) | Mean-field Gaussian variational posterior, ELBO, prior N(0, 1) | 50 weight samples, averaged |

Data: MNIST, 60,000 train / 10,000 test. The training set is split into 54,000 train + 6,000 validation (validation only for hyperparameter selection); the test set is used once for final evaluation. Seed 42, CPU, a few minutes to reproduce.

## Results (clean test set, n = 10,000)

| Model | Accuracy | NLL | ECE |
| --- | --- | --- | --- |
| Standard network (MAP) | 97.23% | 0.125 | **0.88%** |
| MC Dropout | **97.43%** | **0.093** | 3.30% |
| Bayes by Backprop | 94.94% | 0.184 | 6.10% |

The headline finding cuts against the common expectation that "Bayesian = better calibrated": **the standard network has the lowest calibration error**, while probability-averaged predictions from MC Dropout and BBB are systematically *underconfident*. The measurable advantage of the Bayesian methods is in the **uncertainty signal**:

- **Under input noise** (Gaussian σ 0 → 0.5), mean predictive entropy rises from 0.03 → 0.60 (MAP), 0.14 → 1.06 (MC Dropout), 0.30 → 2.06 (BBB).
- **On pure-noise out-of-distribution inputs**, entropy reaches 1.76 (MAP), 1.94 (MC Dropout), and **2.30 (BBB)** — the theoretical maximum for 10 classes (log 10 ≈ 2.30). The Bayesian models say "I don't know"; the point estimate does not.
- **With only 5,000 training examples**, MAP (93.52%, ECE 2.61%) and MC Dropout (92.65%, ECE 5.55%) stay robust, but BBB drops to 87.72% with ECE 20.13% — its prior is too strong for the data size even after narrowing it to N(0, 0.3). Prior strength must match the data scale.

## How to run

```bash
pip install -r requirements.txt
python3 bayesian_mnist_experiments.py
```

MNIST downloads automatically into `data/` on the first run. The script is deterministic (fixed seed 42) and produces:

- `results/results.json` — all numeric results behind the report's tables and figures
- `results/fig1_train_curves.png` — training accuracy curves
- `results/fig2_reliability.png` — reliability diagrams on the clean test set
- `results/fig3_entropy_corruption.png` — predictive entropy under noise and OOD inputs
- `results/fig4_small_data.png` — accuracy and ECE at 54,000 vs. 5,000 training examples
- `results/model_*.pt` — trained model checkpoints

## Repo layout

```
├── README.md                        # this file
├── REPORT.md                        # full report (English): literature review, methods, results, discussion
├── bayesian_mnist_experiments.py    # all experiments: full-data, corruption/OOD, small-data
└── requirements.txt                 # torch, torchvision, numpy, matplotlib
```

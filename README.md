# MNIST Classification with a Fully Connected Neural Network

A PyTorch-based project exploring neural network training, seed robustness, validation-based model selection, hyperparameter tuning, and learned feature visualization on the MNIST handwritten digit dataset.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Setup](#setup)
- [Questions & Results](#questions--results)
  - [Q1 – Baseline Training](#q1--baseline-training)
  - [Q2 – Seed Robustness](#q2--seed-robustness)
  - [Q3 – Validation-Based Model Selection](#q3--validation-based-model-selection)
  - [Q4 – Hyperparameter Tuning](#q4--hyperparameter-tuning)
  - [Q5 – t-SNE Feature Visualization](#q5--t-sne-feature-visualization)

---

## Overview

This project implements a **2-layer fully connected neural network** for classifying MNIST digits (0–9). The experiments progressively build on each other:

| Question | Topic | Goal |
|----------|-------|------|
| Q1 | Baseline training | Train, evaluate, and identify misclassified images |
| Q2 | Seed analysis | Measure variance across 5 random seeds |
| Q3 | Validation set | Use a held-out validation set for early stopping |
| Q4 | Grid search | Tune batch size, hidden size, and learning rate |
| Q5 | t-SNE visualization | Compare raw inputs vs. learned hidden representations |

---

## Architecture

```
Input (784) ──▶ Linear (784 → 500) ──▶ ReLU ──▶ Linear (500 → 10) ──▶ Output
```

| Component | Detail |
|-----------|--------|
| Input size | 784 (28 × 28 flattened) |
| Hidden size | 500 (baseline) |
| Output classes | 10 |
| Activation | ReLU |
| Loss | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 5 |
| Batch size | 100 |
| Training data | 10% of MNIST (~6 000 samples) |

---

## Setup

### Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy
- pandas
- scikit-learn

### Install dependencies

```bash
pip install torch torchvision matplotlib numpy pandas scikit-learn
```

### Run

```bash
python code.py
```

> MNIST data will be automatically downloaded to `./data/` on the first run.

---

## Questions & Results

### Q1 – Baseline Training

The network was trained for **5 epochs** using the hyperparameters listed above. This was done by replicating the code shown in class, using the same hyperparameters.

**What the code produces:**

- **Train & Test loss curves** — a line plot showing how both losses decrease over epochs.

```
Epoch [1/5], Train Loss: X.XXXX, Train Acc: XX.XX%, Test Loss: X.XXXX, Test Acc: XX.XX%
Epoch [2/5], ...
...
```

- **Misclassified images grid** — a 2×5 grid showing 10 test images the model got wrong, along with the true and predicted labels.

| | | | | |
|---|---|---|---|---|
| True: 5 / Pred: 3 | True: 4 / Pred: 9 | ... | ... | ... |
| True: 7 / Pred: 1 | True: 2 / Pred: 8 | ... | ... | ... |

*(Actual digits and predictions vary by run)*

---

### Q2 – Seed Robustness

The full training pipeline from Q1 was repeated with **5 different seeds**: `[2, 13, 56, 100, 523]`.

**Key finding:** The variance was low across all runs. This tells us the model is **robust to the choice of seed** — the random initialization does not significantly affect the final performance.

**What the code produces:**

- **Test error curves** for each seed overlaid on a single plot.
- **Statistics** printed to console:

| Metric | Value |
|--------|-------|
| Mean test error | Computed across 5 seeds |
| Std deviation | Low value → consistent performance |
| Variance | Low value → seed-robust model |

---

### Q3 – Validation-Based Model Selection

A **validation set of 1 000 samples** was randomly sampled from the training data. For each of the 5 seeds, the epoch with the **minimum validation error** was selected, and the corresponding test error was reported.

**What the code produces:**

- Per-seed table:

| Seed | Min Validation Error | Corresponding Test Error |
|------|---------------------|-------------------------|
| 2 | ... | ... |
| 13 | ... | ... |
| 56 | ... | ... |
| 100 | ... | ... |
| 523 | ... | ... |

- **Combined plot** of test and validation error curves for every seed.
- The seed/epoch combination with the **overall lowest validation error** and its corresponding test error.

---

### Q4 – Hyperparameter Tuning

A **grid search** was performed over three hyperparameters:

| Hyperparameter | Values |
|---------------|--------|
| Batch size | 100, 200, 300 |
| Hidden size | 500, 1000, 1500 |
| Learning rate | 0.001, 0.01, 0.1 |

This yields **27 combinations** in total. For each, the model was trained and evaluated using the validation procedure from Q3.

**What the code produces:**

- A **results table** (pandas DataFrame) with all 27 combinations, their validation error, and corresponding test error.
- The **best combination** is highlighted — the one with the lowest validation error.

| Batch Size | Hidden Size | Learning Rate | Validation Error | Test Error |
|-----------|-------------|---------------|-----------------|------------|
| 100 | 500 | 0.001 | ... | ... |
| 100 | 500 | 0.01 | ... | ... |
| ... | ... | ... | ... | ... |

---

### Q5 – t-SNE Feature Visualization

Two **t-SNE** (t-distributed Stochastic Neighbor Embedding) plots are generated side-by-side:

1. **t-SNE of hidden features z_i** — the output of the first linear layer + ReLU
2. **t-SNE of original inputs x_i** — the raw 784-dimensional pixel vectors

Each point is color-coded by its digit class (0–9).

**Key finding:**

> The most significant difference between the plots is that in the **x_i plot** there is noticeable **overlapping between clusters** and a higher number of points in the wrong cluster. In the **z_i plot**, there is a much more **distinct separation** between each cluster and far fewer misplaced points.

This tells us the first layer of the network **successfully learns features useful for classification**, as shown by the clear cluster separation in the z_i embedding.

| x_i (raw input) | z_i (learned features) |
|:---:|:---:|
| Overlapping clusters | Well-separated clusters |
| Harder to distinguish digits | Clear digit boundaries |

---

## Project Structure

```
code (5)/
├── code.py        # All experiments (Q1–Q5)
├── README.md      # This file
└── data/          # MNIST dataset (auto-downloaded)
```

---

## License

This project was created for educational purposes as part of a deep learning course assignment.

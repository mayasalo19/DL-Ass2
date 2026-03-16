# MNIST Classification with a Fully Connected Neural Network

A PyTorch-based project exploring neural network training, seed robustness, validation-based model selection, hyperparameter tuning, and learned feature visualization on the MNIST handwritten digit dataset.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Setup](#setup)
- [Results](#results)
  - [Q1 – Baseline Training](#q1--baseline-training)
  - [Q2 – Seed Robustness](#q2--seed-robustness)
  - [Q3 – Validation-Based Model Selection](#q3--validation-based-model-selection)
  - [Q4 – Hyperparameter Tuning](#q4--hyperparameter-tuning)
  - [Q5 – t-SNE Feature Visualization](#q5--t-sne-feature-visualization)

---

## Overview

This project implements a **2-layer fully connected neural network** for classifying MNIST digits (0–9). The experiments progressively build on each other:

| Question | Topic | Goal |
|:--------:|-------|------|
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
| Training data | 10 % of MNIST (~6 000 samples) |

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
> All figures are saved to the `figures/` directory.

---

## Results

### Q1 – Baseline Training

The network was trained for **5 epochs** using the baseline hyperparameters listed above. This was done by replicating the code shown in class, using the same hyperparameters.

#### Train & Test Loss

<p align="center">
  <img src="figures/q1_train_test_loss.png" alt="Q1 – Train and Test Loss" width="600"/>
</p>

Both train and test loss decrease steadily over the 5 training epochs, indicating the model is learning without significant overfitting.

#### Misclassified Images

<p align="center">
  <img src="figures/q1_misclassified.png" alt="Q1 – Misclassified Images" width="650"/>
</p>

A 2×5 grid of test images the model predicted incorrectly, shown with their true and predicted labels. These tend to be ambiguous or poorly written digits.

---

### Q2 – Seed Robustness

The full training pipeline from Q1 was repeated with **5 different seeds**: `[2, 13, 56, 100, 523]`.

#### Test Error Across Seeds

<p align="center">
  <img src="figures/q2_seed_robustness.png" alt="Q2 – Seed Robustness" width="600"/>
</p>

| Metric | Description |
|--------|-------------|
| **Mean test error** | Computed across 5 seeds |
| **Std deviation** | Low — consistent performance |
| **Variance** | Low — seed-robust model |

**Key finding:** The variance was low across all runs. From this we can conclude that our model is **robust to the choice of seeds** — the random initialization does not significantly affect the final performance.

---

### Q3 – Validation-Based Model Selection

A **validation set of 1 000 samples** was randomly sampled from the training data. For each of the 5 seeds, the epoch with the **minimum validation error** was selected, and the corresponding test error was reported.

#### Test & Validation Error Curves

<p align="center">
  <img src="figures/q3_test_validation_errors.png" alt="Q3 – Test and Validation Errors" width="650"/>
</p>

#### Per-Seed Summary

| Seed | Min Validation Error | Corresponding Test Error |
|:----:|:--------------------:|:------------------------:|
| 2 | — | — |
| 13 | — | — |
| 56 | — | — |
| 100 | — | — |
| 523 | — | — |

> The seed/epoch combination with the **overall lowest validation error** and its corresponding test error is printed to the console.

---

### Q4 – Hyperparameter Tuning

A **grid search** was performed over three hyperparameters:

| Hyperparameter | Values |
|:--------------:|--------|
| Batch size | 100, 200, 300 |
| Hidden size | 500, 1 000, 1 500 |
| Learning rate | 0.001, 0.01, 0.1 |

This yields **27 combinations** in total. For each combination, the model was trained and evaluated using the validation procedure from Q3.

#### Results Table (excerpt)

| Batch Size | Hidden Size | Learning Rate | Validation Error | Test Error |
|:----------:|:-----------:|:-------------:|:----------------:|:----------:|
| 100 | 500 | 0.001 | — | — |
| 100 | 500 | 0.01 | — | — |
| … | … | … | … | … |

> The complete 27-row table and the **best hyperparameter combination** (lowest validation error) are printed when running the code.

---

### Q5 – t-SNE Feature Visualization

Two **t-SNE** (t-distributed Stochastic Neighbor Embedding) plots compare the internal representations learned by the network:

1. **Hidden features z_i** — output of the first linear layer + ReLU  
2. **Original inputs x_i** — raw 784-dimensional pixel vectors

<p align="center">
  <img src="figures/q5_tsne.png" alt="Q5 – t-SNE Visualization" width="750"/>
</p>

#### Analysis

| Aspect | x_i (raw input) | z_i (learned features) |
|--------|:----------------:|:----------------------:|
| Cluster separation | Overlapping clusters | Well-separated clusters |
| Misplaced points | Higher count | Significantly fewer |
| Interpretability | Harder to distinguish digits | Clear digit boundaries |

**Key finding:** The most significant difference between the plots is that in the **x_i plot** there is noticeable **overlapping between clusters** and a higher number of points in the wrong cluster. In the **z_i plot**, there is a much more **distinct separation** between each cluster and far fewer misplaced points.

This tells us the first layer of the network **successfully learns features useful for classification**, as demonstrated by the clear cluster separation in the z_i embedding.

---

## License

This project was created for educational purposes as part of a deep learning course assignment.

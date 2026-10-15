# ML Assignment 4: Ensemble Learning

This repository contains the fourth machine learning assignment, focusing on **ensemble learning for multiclass cyberattack classification** using the dataset from Assignment 2.

Here, **Random Forest** is implemented and compared against the Assignment 2 reference models, *k*-nearest neighbours and classification trees.

---

## Dataset Overview

- **Total observations:** 257,673 across 10 classes
- **Tuning set:** 51,534 observations
- **Final evaluation set:** 206,139 observations

> **Note:** `networkTraffic.csv` is excluded from Git tracking. Please place it inside the `data/` directory before running the code.

---

## Final Random Forest Configuration

- **Number of trees:** 200
- **Criterion:** Entropy
- **Max depth:** `None`
- **Minimum leaf size:** 1
- **Max features:** 0.5
- **Sampling:** Bootstrap sampling
- **Class weighting:** Balanced
- **Random state:** 42

---

## Final Performance Results

| Model | Macro F1 | Accuracy |
| :--- | :---: | :---: |
| **k-nearest neighbours** | 0.4799 | 0.7795 |
| **Decision Tree** | 0.5650 | **0.8046** |
| **Random Forest** | **0.5896** | 0.7735 |

---

## Repository Structure

```text
├── data/              # Dataset and supporting files
├── docs/              # Local working documents, excluded from Git
├── results/           # Saved experimental results and outputs
├── src/               # Python scripts for model training and evaluation
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Place `networkTraffic.csv` inside the `data/` directory, then run the final evaluation with:

```bash
python src/final_evaluation.py
```
# Week 7 — Ensemble Learning for Breast Cancer Diagnosis

**Objective:** Implement and compare Bagging and Boosting to improve accuracy and stability for a breast‑cancer diagnosis task.

## Dataset
- **Source:** `sklearn.datasets.load_breast_cancer()` (Wisconsin Diagnostic Breast Cancer)
- **Docs:** https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset

## Environment Setup (VS Code + GitHub)
```bash
# clone your private repo, then in the repo root:
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

## How to Run (clean console output)
```bash
# run all experiments (baseline, bagging, boosting) and save figures
python -m src.main --all

# or run individual pieces
python -m src.main --baseline
python -m src.main --bagging
python -m src.main --boosting

# optional: choose CV folds and random seed
python -m src.main --all --cv 5 --seed 42
```

Figures (confusion matrices and staged curves) will be written to `figures/`.

## What’s Implemented
- Reproducible train/test split (fixed `random_state`)
- `Pipeline` to avoid leakage (fit transforms on train only)
- Baseline **DecisionTreeClassifier**
- **BaggingClassifier** (with OOB)
- **GradientBoostingClassifier** (Boosting)
- Metrics: accuracy, precision, recall, F1 (macro), confusion matrix
- Stability via **StratifiedKFold CV** (mean ± std on training fold)
- Clear terminal tables + saved plots

## Responsible Modeling Guardrails
- Public, de-identified data
- Fixed split + seeds documented
- Train-only fitting for transforms
- Brief clinical interpretation printed to console

## Repo Layout
```
<repo-root>/
  README.md
  requirements.txt
  .gitignore
  src/
    data_load.py
    preprocess.py
    base_tree.py
    bagging.py
    boosting.py
    evaluate.py
    main.py
  figures/
```


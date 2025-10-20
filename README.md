# Week 7 — Enhancing Predictive Models with Bagging and Boosting  
---

## Project Overview
This project explores ensemble learning techniques to improve diagnostic accuracy and prediction stability in healthcare applications.  
The focus is on **Bagging** and **Boosting** methods using the **Wisconsin Diagnostic Breast Cancer** dataset.  
By comparing a single Decision Tree baseline against ensemble methods, the project demonstrates how variance and bias can be controlled to build more reliable AI-assisted diagnostic tools.

---

## Objectives
- Implement Bagging and Boosting to enhance performance and stability.  
- Evaluate models using accuracy, precision, recall, F1-score, and cross-validation.  
- Analyze improvements in predictive reliability for breast cancer diagnosis.  
- Interpret results in a clinical context, emphasizing recall and diagnostic trust.

---

## Dataset
- **Source:** [`sklearn.datasets.load_breast_cancer`](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)  
- **Samples:** 569 patient records  
- **Features:** 30 numeric variables describing cell nuclei measurements  
- **Target:** 0 = benign, 1 = malignant (re-labeled for clinical clarity)  
- **Privacy:** Fully de-identified, public dataset  

---

## Methodology
| Step | Description |
|------|--------------|
| **Baseline** | Decision Tree Classifier trained as reference model |
| **Bagging** | 50 Decision Trees combined via bootstrapped sampling with OOB scoring |
| **Boosting** | Gradient Boosting Classifier (200 estimators, learning rate = 0.05, max depth = 2) |
| **Pipeline** | ColumnTransformer + Imputer ensures clean preprocessing and no data leakage |
| **Evaluation** | Accuracy, Precision, Recall, F1-macro, Confusion Matrix, and Stratified K-Fold CV |

---

##  Results Summary
| Model | Accuracy | Precision (M) | Recall (M) | F1 (M) | CV F1 ± std |
|:------|:---------:|:-------------:|:----------:|:------:|:------------:|
| Decision Tree | 0.93 | 0.92 | 0.92 | 0.92 | 0.918 ± 0.035 |
| **Bagging** | **0.97** | **0.98** | **0.96** | **0.97** | **0.958 ± 0.020** |
| Boosting | 0.96 | 0.97 | 0.95 | 0.96 | 0.965 ± 0.015 |

**Key Insight:**  
Bagging achieved the highest accuracy and variance reduction, while Boosting delivered the most consistent cross-validation stability.  
Both ensembles significantly improved recall, reducing the likelihood of missed malignant cases.

---

## Figures
- `figures/cm_baseline.png` — Baseline Decision Tree Confusion Matrix  
- `figures/cm_bagging.png` — Bagging Confusion Matrix  
- `figures/cm_boosting.png` — Gradient Boosting Confusion Matrix  
- `figures/boosting_staged_f1.png` — Boosting staged F1 curve  

---

## Clinical Interpretation
- Higher recall translates to fewer undetected malignant cases, supporting early intervention.  
- Bagging improves reliability through variance reduction.  
- Boosting refines predictions by correcting residual errors sequentially.  
- Trade-offs include longer training time and reduced interpretability compared to a single tree.  

---

## How to Run
```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\Activate.ps1    # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Run all experiments
python -m src.main --all

# Or run individually
python -m src.main --baseline
python -m src.main --bagging
python -m src.main --boosting

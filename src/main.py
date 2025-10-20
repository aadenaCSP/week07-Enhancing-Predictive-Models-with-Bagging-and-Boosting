from __future__ import annotations
import argparse, os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_load import load_wdbc
from src.preprocess import build_preprocessor
from src.base_tree import build_base_tree
from src.bagging import build_bagging
from src.boosting import build_boosting
from src.evaluate import compute_metrics, print_metrics_table, save_confusion_matrix, cv_stability, plot_staged_curve

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

def run_baseline(X, y, feature_names, test_size: float, seed: int, cv_folds: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    pre, _ = build_preprocessor(feature_names)
    model = build_base_tree(random_state=seed, max_depth=None, min_samples_leaf=1)
    pipe = Pipeline([("pre", pre), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    print_metrics_table("Baseline Decision Tree (Test)", metrics)
    mean_cv, std_cv = cv_stability(pipe, X_train, y_train, cv_folds=cv_folds, scoring="f1_macro", random_state=seed)
    print(f"CV f1_macro (train folds): {mean_cv:.4f} ± {std_cv:.4f}")
    save_confusion_matrix(y_test, y_pred, labels=["benign","malignant"], out_path=os.path.join(FIG_DIR, "cm_baseline.png"), title="Baseline Tree — Confusion Matrix")
    return {"name": "Baseline Tree", "metrics": metrics, "cv_mean": mean_cv, "cv_std": std_cv}

def run_bagging(X, y, feature_names, test_size: float, seed: int, cv_folds: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    pre, _ = build_preprocessor(feature_names)
    base = build_base_tree(random_state=seed, max_depth=None, min_samples_leaf=1)
    bag = build_bagging(base_estimator=base, n_estimators=50, bootstrap=True, oob_score=True, random_state=seed)
    pipe = Pipeline([("pre", pre), ("clf", bag)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    print_metrics_table("Bagging (Test)", metrics)
    clf = pipe.named_steps["clf"]
    if getattr(clf, "oob_score_", None) is not None:
        print(f"OOB Score (train): {clf.oob_score_:.4f}")
    mean_cv, std_cv = cv_stability(pipe, X_train, y_train, cv_folds=cv_folds, scoring="f1_macro", random_state=seed)
    print(f"CV f1_macro (train folds): {mean_cv:.4f} ± {std_cv:.4f}")
    save_confusion_matrix(y_test, y_pred, labels=["benign","malignant"], out_path=os.path.join(FIG_DIR, "cm_bagging.png"), title="Bagging — Confusion Matrix")
    return {"name": "Bagging", "metrics": metrics, "cv_mean": mean_cv, "cv_std": std_cv, "oob": getattr(clf, "oob_score_", None)}

def run_boosting(X, y, feature_names, test_size: float, seed: int, cv_folds: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    pre, _ = build_preprocessor(feature_names)
    gb = build_boosting(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=seed)
    pipe = Pipeline([("pre", pre), ("clf", gb)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    print_metrics_table("Gradient Boosting (Test)", metrics)
    mean_cv, std_cv = cv_stability(pipe, X_train, y_train, cv_folds=cv_folds, scoring="f1_macro", random_state=seed)
    print(f"CV f1_macro (train folds): {mean_cv:.4f} ± {std_cv:.4f}")
    save_confusion_matrix(y_test, y_pred, labels=["benign","malignant"], out_path=os.path.join(FIG_DIR, "cm_boosting.png"), title="Gradient Boosting — Confusion Matrix")
    # staged curve on transformed training data
    X_tr = pipe.named_steps["pre"].fit_transform(X_train, y_train)
    plot_staged_curve(pipe.named_steps["clf"], X_tr, y_train, out_path=os.path.join(FIG_DIR, "boosting_staged_f1.png"), metric="f1_macro")
    return {"name": "Gradient Boosting", "metrics": metrics, "cv_mean": mean_cv, "cv_std": std_cv}

def clinical_note():
    print("""
--- Clinical interpretation (brief) ---
Higher recall macro indicates fewer missed malignant cases, which aligns with safety goals.
Bagging reduces variance relative to a single decision tree, leading to steadier performance.
Boosting often increases overall accuracy and recall by correcting previous errors sequentially.
Trade-offs include extra training/tuning time and reduced model interpretability versus a single tree.
""")

def main():
    import sys
    parser = argparse.ArgumentParser(description="Week 7 — Ensemble Learning for Breast Cancer Diagnosis")
    parser.add_argument("--baseline", action="store_true", help="Run baseline Decision Tree only")
    parser.add_argument("--bagging", action="store_true", help="Run Bagging only")
    parser.add_argument("--boosting", action="store_true", help="Run Gradient Boosting only")
    parser.add_argument("--all", action="store_true", help="Run baseline + bagging + boosting")
    parser.add_argument("--cv", type=int, default=5, help="CV folds for stability check (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction (default: 0.2)")
    args = parser.parse_args()

    if not any([args.baseline, args.bagging, args.boosting, args.all]):
        args.all = True  # sensible default

    X, y, feature_names = load_wdbc()

    results = []
    if args.all or args.baseline:
        results.append(run_baseline(X, y, feature_names, args.test_size, args.seed, args.cv))
    if args.all or args.bagging:
        results.append(run_bagging(X, y, feature_names, args.test_size, args.seed, args.cv))
    if args.all or args.boosting:
        results.append(run_boosting(X, y, feature_names, args.test_size, args.seed, args.cv))

    print("\n=== Comparison (Test Metrics) ===")
    header = f"{'Model':<20} {'Acc':>6} {'Prec(M)':>8} {'Rec(M)':>8} {'F1(M)':>8} {'CV F1 ± std':>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        m = r['metrics']
        cv_str = f"{r['cv_mean']:.3f} ± {r['cv_std']:.3f}"
        print(f"{r['name']:<20} {m['accuracy']:>6.3f} {m['precision_macro']:>8.3f} {m['recall_macro']:>8.3f} {m['f1_macro']:>8.3f} {cv_str:>14}")

    clinical_note()

if __name__ == "__main__":
    main()
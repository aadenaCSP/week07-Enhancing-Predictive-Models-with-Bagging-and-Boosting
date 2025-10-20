from typing import Tuple, List
import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_wdbc() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load the Wisconsin Diagnostic Breast Cancer dataset from scikit-learn.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels (0/1); 1 indicates malignant in this script for recall focus.
    feature_names : list of str
        Names of numeric features.
    """
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=[str(c) for c in ds.feature_names])  # <- coerce to str
    # sklearn: 0 = malignant, 1 = benign. Flip so 1 = malignant for recall focus.
    y = pd.Series((ds.target == 0).astype(int), name="malignant")
    feature_names = [str(c) for c in X.columns]  # <- plain Python str
    return X, y, feature_names

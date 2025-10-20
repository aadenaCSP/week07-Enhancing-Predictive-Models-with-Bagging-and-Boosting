from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def build_preprocessor(numeric_features: List[str]) -> Tuple[ColumnTransformer, Pipeline]:
    """
    Build a preprocessing ColumnTransformer and a wrapper Pipeline.

    Notes
    -----
    - Dataset is numeric and has no missing values; we still include an imputer
      (median) for robustness and to demonstrate a clean train-only fit.
    - No scaling is applied because the primary models are tree-based.
    """
    # Ensure plain Python str column names (not numpy.str_)
    numeric_features = [str(c) for c in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # Add StandardScaler() here later if you introduce a scale-sensitive comparator.
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor, Pipeline(steps=[("pre", preprocessor)])

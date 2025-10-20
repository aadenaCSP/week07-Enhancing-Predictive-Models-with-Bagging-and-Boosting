from sklearn.ensemble import BaggingClassifier
from sklearn.base import BaseEstimator

def build_bagging(base_estimator: BaseEstimator,
                  n_estimators: int = 50,
                  bootstrap: bool = True,
                  oob_score: bool = True,
                  random_state: int = 42) -> BaggingClassifier:
    """
    Build a Bagging ensemble around the provided base estimator.
    """
    return BaggingClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        bootstrap=bootstrap,
        oob_score=oob_score,
        random_state=random_state,
        n_jobs=-1
    )
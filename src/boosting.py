from sklearn.ensemble import GradientBoostingClassifier

def build_boosting(n_estimators: int = 200,
                   learning_rate: float = 0.05,
                   max_depth: int = 2,
                   random_state: int = 42) -> GradientBoostingClassifier:
    """
    Build a GradientBoostingClassifier (tree-based boosting).
    """
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=1.0,
        random_state=random_state
    )
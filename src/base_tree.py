from sklearn.tree import DecisionTreeClassifier

def build_base_tree(random_state: int = 42,
                    max_depth: int | None = None,
                    min_samples_leaf: int = 1) -> DecisionTreeClassifier:
    """
    Create a baseline DecisionTreeClassifier.
    Allowing moderate depth exposes variance, which Bagging can mitigate.
    """
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
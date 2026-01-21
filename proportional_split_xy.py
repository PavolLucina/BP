import numpy as np
import pandas as pd

def proportional_split_xy(X, y, query_ratio=0.2):
    """
    Proportionally split X, y into train/test while preserving class distributions per identity.
    
    Parameters:
        X : np.ndarray or pd.DataFrame
            Feature matrix (N x D)
        y : np.ndarray or pd.Series
            Labels (N,)
        query_ratio : float
            Fraction of each identity to include in the test set.

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Make a DataFrame for easy grouping
    df = pd.DataFrame({
        "idx": np.arange(len(y)),
        "identity": y
    })

    train_indices = []
    test_indices = []

    for _, group in df.groupby("identity"):
        num_images = len(group)
        num_test = max(1, int(np.ceil(num_images * query_ratio)))
        shuffled = group.sample(frac=1, random_state=42)

        test_indices.extend(shuffled.iloc[:num_test]["idx"].tolist())
        train_indices.extend(shuffled.iloc[num_test:]["idx"].tolist())

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

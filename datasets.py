import numpy as np
from typing import Tuple, Dict, Any

import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_mnist(
    validation_split: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load MNIST from tf.keras.datasets, normalize to [0,1], and split into train/val/test.
    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test, metadata
    """
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize and add channel dimension
    x_train_full = (x_train_full.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    # train/validation split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=validation_split, stratify=y_train_full, random_state=42
    )

    num_classes = 10
    metadata = {
        "num_classes": num_classes,
        "input_shape": x_train.shape[1:],
        "train_size": x_train.shape[0],
    }

    # ============================================================
    # DEBUG MODE — reduce dataset size for faster DP-SGD training
    # ============================================================
    x_train = x_train[:5000]
    y_train = y_train[:5000]

    x_val = x_val[:1000]
    y_val = y_val[:1000]

    x_test = x_test[:1000]
    y_test = y_test[:1000]

    # Update metadata to match reduced dataset
    metadata["train_size"] = x_train.shape[0]
    metadata["input_shape"] = x_train.shape[1:]
    # ============================================================

    return x_train, y_train, x_val, y_val, x_test, y_test, metadata


def _load_adult_dataframe():
    """
    Try to load Adult Income from OpenML (requires internet).
    Returns a pandas DataFrame or raises an error if unavailable.
    """
    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame
    df.columns = [c.strip() for c in df.columns]
    return df


def _preprocess_adult(df):
    """
    Preprocess Adult Income DataFrame into numeric numpy arrays with one-hot encoding for categorical variables.
    Label: income >50K ?
    """
    import pandas as pd

    target_col = "class" if "class" in df.columns else "income"
    y = (df[target_col].astype(str).str.contains(">50K")).astype(int).values

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    cat_cols = [c for c in df.columns if c not in numeric_cols + [target_col]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    X = preprocessor.fit_transform(df)
    X = X.astype("float32")

    # If sparse, convert to dense
    if hasattr(X, "toarray"):
        X = X.toarray()

    return X, y


def load_adult(
    validation_size: float = 0.1,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load and preprocess the UCI Adult Income dataset from OpenML, then split into train/val/test.
    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test, metadata
    """
    df = _load_adult_dataframe()
    X, y = _preprocess_adult(df)

    # First split into train/test
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    # Then split train into train/validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=validation_size, stratify=y_train_full, random_state=42
    )

    num_classes = 2
    metadata = {
        "num_classes": num_classes,
        "input_shape": (x_train.shape[1],),
        "train_size": x_train.shape[0],
    }
    return x_train, y_train, x_val, y_val, x_test, y_test, metadata

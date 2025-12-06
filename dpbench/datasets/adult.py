import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torch

def _load_openml():
    try:
        from sklearn.datasets import fetch_openml
        adult = fetch_openml('adult', version=2, as_frame=True)
        df = adult.frame
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return None

def _load_local_csv(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def _preprocess(df):
    target_col = 'class' if 'class' in df.columns else 'income'
    y = (df[target_col].astype(str).str.contains('>50K')).astype(int).values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols: numeric_cols.remove(target_col)
    cat_cols = [c for c in df.columns if c not in numeric_cols + [target_col]]
    pre = ColumnTransformer([
        ('num', StandardScaler(with_mean=False), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ])
    X = pre.fit_transform(df).astype('float32')
    return X, y

def get_adult(root='./data', csv_name='adult.csv', test_size=0.2, val_size=0.2, seed=42):
    os.makedirs(root, exist_ok=True)
    df = _load_openml() or _load_local_csv(os.path.join(root, csv_name))
    if df is None:
        raise FileNotFoundError('Adult not available: no internet and no data/adult.csv')
    X, y = _preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train)
    toT = lambda a: torch.tensor(a.toarray() if hasattr(a, 'toarray') else a)
    train_ds = TensorDataset(toT(X_train), torch.tensor(y_train).long())
    val_ds   = TensorDataset(toT(X_val),   torch.tensor(y_val).long())
    test_ds  = TensorDataset(toT(X_test),  torch.tensor(y_test).long())
    input_dim = train_ds.tensors[0].shape[1]
    return train_ds, val_ds, test_ds, input_dim

"""Fit scaling on training rows and resample training classes only."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from .data import NUMERIC_COLUMNS


def scale_split(train, validation):
    scaler = StandardScaler()
    train_scaled = train.copy()
    validation_scaled = validation.copy()
    train_scaled[NUMERIC_COLUMNS] = scaler.fit_transform(train[NUMERIC_COLUMNS])
    validation_scaled[NUMERIC_COLUMNS] = scaler.transform(validation[NUMERIC_COLUMNS])
    return train_scaled, validation_scaled, scaler


def oversample(train, target, seed=42):
    if not train.index.equals(target.index):
        raise ValueError("Training features and target must have aligned indices.")
    table = train.copy()
    table["popularity"] = target
    target_count = target.value_counts().max()
    groups = []
    for _, group in table.groupby("popularity"):
        if len(group) < target_count:
            extra = resample(group, replace=True, n_samples=target_count-len(group),
                             random_state=seed)
            group = pd.concat([group, extra])
        groups.append(group)
    balanced = pd.concat(groups).sample(frac=1, random_state=seed).reset_index(drop=True)
    return balanced.drop(columns="popularity"), balanced["popularity"]

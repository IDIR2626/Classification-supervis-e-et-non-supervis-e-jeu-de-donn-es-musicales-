"""Read the source file and validate the assumptions used in the experiments."""
from pathlib import Path
import hashlib
import pandas as pd

NUMERIC_COLUMNS = [
    "danceability", "energy", "loudness", "acousticness", "speechiness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]
FEATURE_COLUMNS = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms", "explicit",
]
TARGET = "popularity"
METADATA_COLUMNS = ["track_name", "track_artist", "track_id"]


def load_data(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. See data/README.md for download instructions."
        )
    df = pd.read_csv(path)
    required = FEATURE_COLUMNS + [TARGET] + METADATA_COLUMNS
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Missing values found. Define an imputation strategy before running.")
    if not df["explicit"].isin([True, False, 0, 1]).all():
        raise ValueError("explicit must contain boolean values or 0/1.")
    X = df[FEATURE_COLUMNS].copy()
    X["explicit"] = X["explicit"].astype(int)
    for column in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(X[column]):
            raise ValueError(f"Expected a numeric feature: {column}")
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        raise ValueError("popularity must be numeric.")
    if not (df[TARGET] % 1 == 0).all():
        raise ValueError("Classification expects integer popularity labels.")
    return df, X, df[TARGET].astype(int).copy()


def source_metadata(path, df):
    return {
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "rows": len(df), "columns": len(df.columns),
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "zero_tempos": int(df["tempo"].eq(0).sum()),
    }

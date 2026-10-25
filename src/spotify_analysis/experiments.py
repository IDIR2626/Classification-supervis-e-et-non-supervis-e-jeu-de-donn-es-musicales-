"""Run classification and clustering with explicit validation boundaries."""
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, davies_bouldin_score, silhouette_score, rand_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from .preprocessing import oversample, scale_split


def metrics(y_true, predicted, labels):
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "error": 1-accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "macro_f1": f1_score(y_true, predicted, labels=labels, average="macro", zero_division=0),
    }


def select_columns(train, target, seed):
    tree = DecisionTreeClassifier(max_depth=5, random_state=seed).fit(train, target)
    importance = pd.Series(tree.feature_importances_, index=train.columns).sort_values(ascending=False)
    columns = importance[importance > 0].head(6).index.tolist()
    if not columns:
        raise ValueError("The selector found no discriminating feature.")
    return columns, tree


def classification(X, y, seed=42):
    if y.value_counts().min() < 4:
        raise ValueError("At least four original rows per class are needed for this protocol.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.30, random_state=seed, stratify=y)
    if y_train.value_counts().min() < 3:
        raise ValueError("At least three training rows per class are needed for three folds.")
    labels = sorted(y.unique())
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores, selections = [], []
    confusion_full = np.zeros((len(labels), len(labels)), dtype=int)
    for fold, (it, iv) in enumerate(cv.split(X_train, y_train), 1):
        A, V, _ = scale_split(X_train.iloc[it], X_train.iloc[iv])
        a, v = y_train.iloc[it], y_train.iloc[iv]
        B, b = oversample(A, a, seed)
        columns, selector = select_columns(B, b, seed)
        selections.append({"fold": fold, "features": ", ".join(columns)})

        # This model uses all 13 features; the grid below uses six selected features.
        predicted_full = selector.predict(V)
        confusion_full += confusion_matrix(v, predicted_full, labels=labels)
        # Measure a separate fit call, excluding preprocessing and selection.
        full = DecisionTreeClassifier(max_depth=5, random_state=seed)
        started = perf_counter(); full.fit(B, b); duration = perf_counter()-started
        scores.append({"family": "tree_all_features", "parameter": 5, "fold": fold,
                       "fit_seconds": duration, **metrics(v, predicted_full, labels)})

        baseline = DummyClassifier(strategy="most_frequent")
        started = perf_counter(); baseline.fit(A, a); duration = perf_counter()-started
        scores.append({"family": "majority_baseline", "parameter": 0, "fold": fold,
                       "fit_seconds": duration, **metrics(v, baseline.predict(V), labels)})

        for family, parameters in [("decision_tree", range(1, 6)), ("knn", range(2, 21))]:
            for parameter in parameters:
                model = (DecisionTreeClassifier(max_depth=parameter, random_state=seed)
                         if family == "decision_tree" else KNeighborsClassifier(n_neighbors=parameter))
                started = perf_counter(); model.fit(B[columns], b); duration = perf_counter()-started
                scores.append({"family": family, "parameter": parameter, "fold": fold,
                               "fit_seconds": duration,
                               **metrics(v, model.predict(V[columns]), labels)})
    scores = pd.DataFrame(scores)
    summary = scores.groupby(["family", "parameter"], as_index=False)[
        ["accuracy", "error", "balanced_accuracy", "macro_f1", "fit_seconds"]].mean()
    # Choose among tree/k-NN candidates on CV accuracy, before looking at the test.
    candidates = summary[summary.family.isin(["decision_tree", "knn"])]
    best = candidates.sort_values(["accuracy", "family", "parameter"],
                                 ascending=[False, True, True]).iloc[0]
    A, T, _ = scale_split(X_train, X_test)
    B, b = oversample(A, y_train, seed)
    columns, _ = select_columns(B, b, seed)
    model = (DecisionTreeClassifier(max_depth=int(best.parameter), random_state=seed)
             if best.family == "decision_tree" else KNeighborsClassifier(n_neighbors=int(best.parameter)))
    model.fit(B[columns], b)
    predicted = model.predict(T[columns])
    baseline = DummyClassifier(strategy="most_frequent").fit(A, y_train)
    holdout = {
        "selected_family": best.family, "selected_parameter": int(best.parameter),
        "selection_metric": "mean CV accuracy among decision_tree and knn candidates",
        "selected_features": columns, "test_rows": len(y_test),
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "candidate": metrics(y_test, predicted, labels),
        "majority_baseline": metrics(y_test, baseline.predict(T), labels),
    }
    return scores, summary, pd.DataFrame(selections), confusion_full, labels, holdout


def clustering(X, y, seed=42, repeats=10):
    rows = []
    for repetition in range(repeats):
        current_seed = seed+repetition
        A, T, a, t = train_test_split(X, y, test_size=.30, random_state=current_seed)
        A, T, _ = scale_split(A, T)
        km = KMeans(n_clusters=y.nunique(), n_init=10, random_state=current_seed).fit(A)
        assigned = km.predict(T)
        if not 2 <= len(np.unique(assigned)) < len(T):
            raise ValueError(f"Undefined internal scores at repetition {repetition+1}.")
        rows.append({"repetition": repetition+1, "seed": current_seed,
                     "davies_bouldin": davies_bouldin_score(T, assigned),
                     "silhouette": silhouette_score(T, assigned),
                     "rand": rand_score(t, assigned)})
    return pd.DataFrame(rows)

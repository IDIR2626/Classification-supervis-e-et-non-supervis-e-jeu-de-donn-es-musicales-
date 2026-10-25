"""Command-line entry point for a complete reproducible run."""
import argparse
import json
import platform
from pathlib import Path
import pandas as pd
import sklearn
from .data import load_data, source_metadata
from .preprocessing import scale_split, oversample
from .experiments import classification, clustering
from .plots import exploration, comparison, confusion, cluster_scores, descriptive_tree
from sklearn.tree import DecisionTreeClassifier


def main():
    parser = argparse.ArgumentParser(description="Analyze low-popularity Spotify tracks.")
    parser.add_argument("--data", type=Path, default=Path("data/raw/unpopular_songs.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    df, X, y = load_data(args.data)
    figures, tables = args.output/"figures", args.output/"tables"
    figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    exploration(df, figures)
    df.describe(include="all").T.to_csv(tables/"descriptive_statistics.csv")
    df.popularity.value_counts().sort_index().rename("count").to_csv(tables/"class_counts.csv")
    scores, summary, selections, matrix, labels, holdout = classification(X, y, args.seed)
    scores.to_csv(tables/"classification_folds.csv", index=False)
    summary.to_csv(tables/"classification_summary.csv", index=False)
    selections.to_csv(tables/"selected_features.csv", index=False)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(tables/"confusion_depth5.csv")
    comparison(summary, figures); confusion(matrix, labels, figures)
    clustered = clustering(X, y, args.seed, args.repeats)
    clustered.to_csv(tables/"clustering_runs.csv", index=False)
    clustered[["davies_bouldin", "silhouette", "rand"]].agg(["mean", "min", "max"]).to_csv(tables/"clustering_summary.csv")
    cluster_scores(clustered, figures)
    # All-data tree is descriptive only, after candidate selection and test evaluation.
    A, _, _ = scale_split(X, X)
    B, b = oversample(A, y, args.seed)
    tree = DecisionTreeClassifier(max_depth=5, random_state=args.seed).fit(B, b)
    pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False).rename("importance").to_csv(tables/"feature_importances.csv")
    descriptive_tree(tree, X.columns.tolist(), labels, figures)
    metadata = {"dataset": source_metadata(args.data, df), "seed": args.seed,
                "clustering_repeats": args.repeats, "python": platform.python_version(),
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                "protocol": "70/30 stratified classification split; 3-fold CV on training; training-only oversampling; 10 repeated 70/30 clustering splits"}
    (args.output/"run_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    (args.output/"holdout_metrics.json").write_text(json.dumps(holdout, indent=2)+"\n")
    print(summary.to_string(index=False))
    print("\nHeld-out candidate:", holdout["selected_family"], holdout["selected_parameter"])
    print(json.dumps(holdout["candidate"], indent=2))
    print("\nMean clustering scores:")
    print(clustered[["davies_bouldin", "silhouette", "rand"]].mean())
    print("\nResults written to", args.output)


if __name__ == "__main__":
    main()

"""Figures used in the README and the generated results folder."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree
from .data import NUMERIC_COLUMNS


def exploration(df, output):
    counts = df.popularity.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    counts.plot.bar(ax=ax, color="#168b8c")
    ax.set(xlabel="Popularity class", ylabel="Tracks", title="A strongly imbalanced target")
    for i, count in enumerate(counts):
        ax.text(i, count+25, str(count), ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(output/"class_distribution.png", dpi=160); plt.close(fig)
    df[NUMERIC_COLUMNS].hist(bins=25, figsize=(13, 9), edgecolor="white", color="#168b8c")
    plt.tight_layout(); plt.savefig(output/"feature_distributions.png", dpi=160); plt.close()


def comparison(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    baseline = summary[summary.family == "majority_baseline"].iloc[0]
    for ax, family in zip(axes, ["decision_tree", "knn"]):
        table = summary[summary.family == family]
        ax.plot(table.parameter, table.accuracy, "o-", color="#168b8c", label="Accuracy")
        ax.plot(table.parameter, table.balanced_accuracy, "s-", color="#c15f3c", label="Balanced accuracy")
        ax.axhline(baseline.accuracy, color="#4c5963", ls="--", label="Majority baseline accuracy")
        ax.set(xlabel="Max depth" if family == "decision_tree" else "Neighbors",
               ylabel="Score", title="Decision tree" if family == "decision_tree" else "k-NN", ylim=(0, 1))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/"model_comparison.png", dpi=170); plt.close(fig)


def confusion(matrix, labels, output):
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i,j]), ha="center", va="center", fontsize=7,
                    color="white" if matrix[i,j] > matrix.max()/2 else "black")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=labels, yticklabels=labels, xlabel="Predicted class",
           ylabel="True class", title="Depth-5 tree: out-of-fold predictions (13 features)")
    fig.colorbar(image, ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(output/"confusion_matrix.png", dpi=170); plt.close(fig)


def cluster_scores(scores, output):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, metric in zip(axes, ["davies_bouldin", "silhouette", "rand"]):
        ax.plot(scores.repetition, scores[metric], "o-", color="#168b8c")
        ax.axhline(scores[metric].mean(), color="#c15f3c", ls="--", label="Mean")
        ax.set(xlabel="Split", title=metric.replace("_", " ").title())
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/"clustering_scores.png", dpi=170); plt.close(fig)


def descriptive_tree(model, features, labels, output):
    fig, ax = plt.subplots(figsize=(30, 12))
    plot_tree(model, feature_names=features, class_names=[str(x) for x in labels],
              filled=True, rounded=True, ax=ax)
    fig.savefig(output/"descriptive_tree.pdf", bbox_inches="tight"); plt.close(fig)

"""Ensemble showcase — forest, boosting, naive Bayes and voting on seeded blobs."""

import random

from cds.ml import (
    GaussianNaiveBayes,
    GradientBoostingClassifier,
    RandomForestClassifier,
    accuracy,
)
from cds.ml.model_selection import SupervisedModel
from cds.ml.voting import VotingClassifier


def make_dataset(n_per_class: int, seed: int) -> tuple[list[list[float]], list[str]]:
    """Two Gaussian blobs in 2-D: class A around (-1, -1), class B around (1, 1)."""
    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[str] = []
    for _ in range(n_per_class):
        X.append([rng.gauss(-1.0, 0.9), rng.gauss(-1.0, 0.9)])
        y.append("A")
        X.append([rng.gauss(1.0, 0.9), rng.gauss(1.0, 0.9)])
        y.append("B")
    return X, y


def report(name: str, model: SupervisedModel, X_test: list[list[float]], y_test: list[str]) -> None:
    y_pred = [model.predict(row) for row in X_test]
    acc = accuracy(y_test, y_pred)
    print(f"  {name:<28} accuracy = {acc:.3f}")


def main() -> None:
    X, y = make_dataset(n_per_class=60, seed=42)

    indices = list(range(len(X)))
    random.Random(7).shuffle(indices)
    split = 80
    train_idx = indices[:split]
    test_idx = indices[split:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    print(f"=== Ensemble showcase on {len(X_train)} train / {len(X_test)} test samples ===\n")
    print("Individual models:")

    forest = RandomForestClassifier(n_trees=30, max_depth=4, seed=42)
    boost = GradientBoostingClassifier(n_estimators=40, learning_rate=0.3, max_depth=2)
    bayes = GaussianNaiveBayes()
    individuals: list[tuple[str, SupervisedModel]] = [
        ("RandomForest", forest),
        ("GradientBoosting", boost),
        ("GaussianNaiveBayes", bayes),
    ]
    for name, model in individuals:
        model.fit(X_train, y_train)
        report(name, model, X_test, y_test)

    # --- Voting ensembles over fresh members with the same hyperparameters ---
    print("\nVoting ensembles:")
    hard_members: list[tuple[str, SupervisedModel]] = [
        ("forest", RandomForestClassifier(n_trees=30, max_depth=4, seed=42)),
        ("boost", GradientBoostingClassifier(n_estimators=40, learning_rate=0.3, max_depth=2)),
        ("bayes", GaussianNaiveBayes()),
    ]
    soft_members: list[tuple[str, SupervisedModel]] = [
        ("forest", RandomForestClassifier(n_trees=30, max_depth=4, seed=42)),
        ("boost", GradientBoostingClassifier(n_estimators=40, learning_rate=0.3, max_depth=2)),
        ("bayes", GaussianNaiveBayes()),
    ]
    hard_vote = VotingClassifier(hard_members, voting="hard").fit(X_train, y_train)
    soft_vote = VotingClassifier(soft_members, voting="soft").fit(X_train, y_train)
    report("VotingClassifier (hard)", hard_vote, X_test, y_test)
    report("VotingClassifier (soft)", soft_vote, X_test, y_test)


if __name__ == "__main__":
    main()

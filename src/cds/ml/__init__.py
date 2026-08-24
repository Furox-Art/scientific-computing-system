"""Machine Learning module for CDS."""

from cds.ml.boosting import GradientBoostingClassifier
from cds.ml.clustering import KMeans, KMeansResult
from cds.ml.decomposition import PCA, PCAResult
from cds.ml.ensemble import RandomForestClassifier
from cds.ml.linear_models import LinearRegression, LogisticRegression
from cds.ml.metrics import (
    ConfusionMatrixResult,
    Prf,
    accuracy,
    confusion_matrix,
    macro_prf,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_f1,
    r2_score,
    roc_auc,
)
from cds.ml.model_selection import CVResult, cross_val_score, k_fold_indices
from cds.ml.naive_bayes import GaussianNaiveBayes
from cds.ml.neighbors import KNeighborsClassifier, KNeighborsRegressor
from cds.ml.neural import MLP, Layer
from cds.ml.preprocessing import StandardScaler, train_test_split
from cds.ml.tree import DecisionTreeClassifier

__all__ = [
    "CVResult",
    "GaussianNaiveBayes",
    "GradientBoostingClassifier",
    "ConfusionMatrixResult",
    "DecisionTreeClassifier",
    "KMeans",
    "KMeansResult",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "Layer",
    "LinearRegression",
    "LogisticRegression",
    "MLP",
    "PCA",
    "PCAResult",
    "Prf",
    "RandomForestClassifier",
    "StandardScaler",
    "accuracy",
    "confusion_matrix",
    "cross_val_score",
    "k_fold_indices",
    "macro_prf",
    "mean_absolute_error",
    "mean_squared_error",
    "precision_recall_f1",
    "r2_score",
    "roc_auc",
    "train_test_split",
]

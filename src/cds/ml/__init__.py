"""Machine Learning module for CDS."""

from cds.ml.clustering import KMeans, KMeansResult
from cds.ml.decomposition import PCA, PCAResult
from cds.ml.linear_models import LinearRegression, LogisticRegression
from cds.ml.neighbors import KNeighborsClassifier, KNeighborsRegressor
from cds.ml.neural import MLP, Layer
from cds.ml.preprocessing import StandardScaler, train_test_split
from cds.ml.tree import DecisionTreeClassifier

__all__ = [
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
    "StandardScaler",
    "train_test_split",
]

"""
models.py
---------
Model definitions and wrappers for the three ML components:
  1. Decision Tree Classifier  → Crop Recommendation
  2. KMeans Clustering         → Soil Segmentation (KNN-style grouping)
  3. Linear Regression         → Crop Yield Prediction
"""

from sklearn.tree           import DecisionTreeClassifier
from sklearn.cluster        import KMeans
from sklearn.linear_model   import LinearRegression
from sklearn.decomposition  import PCA


# ── 1. Decision Tree Classifier ───────────────────────────────────────────────
def build_decision_tree(random_state: int = 42) -> DecisionTreeClassifier:
    """
    Return a configured (untrained) Decision Tree for crop recommendation.

    Hyper-parameters are chosen to balance depth vs. generalisation:
      max_depth=10 avoids extreme overfitting on the 7-feature space.
      min_samples_leaf=4 reduces noise sensitivity.
    """
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=4,
        random_state=random_state
    )


# ── 2. KMeans Clustering (Soil Segmentation) ──────────────────────────────────
def build_kmeans(n_clusters: int = 5, random_state: int = 42) -> KMeans:
    """
    Return a configured (untrained) KMeans model.

    n_clusters=5 represents five archetypal soil zones.
    n_init=15 ensures stable centroid initialisation.
    """
    return KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=15,
        max_iter=500,
        random_state=random_state
    )


def build_pca(n_components: int = 2) -> PCA:
    """Return a PCA reducer for 2-D cluster visualisation."""
    return PCA(n_components=n_components, random_state=42)


# ── 3. Linear Regression (Yield Prediction) ───────────────────────────────────
def build_linear_regression() -> LinearRegression:
    """
    Return a configured (untrained) Linear Regression model.

    fit_intercept=True lets the model capture a global baseline yield.
    """
    return LinearRegression(fit_intercept=True)

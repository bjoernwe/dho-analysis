import numpy as np

from sklearn.decomposition import PCA


class CustomPCA:

    def __init__(self, n_components: int):
        self._pca_full: PCA = PCA()
        self._pca_reduced: PCA = PCA(n_components=n_components)

    def fit(self, X: np.ndarray):
        self._pca_full.fit(X=X)
        self._pca_reduced.fit(X=X)

    @property
    def n_components_reduced_(self) -> int:
        return self._pca_reduced.n_components_

    @property
    def components_reduced_(self) -> np.ndarray:
        return self._pca_reduced.components_

    @property
    def explained_variance_ratio_full_(self) -> np.ndarray:
        return self._pca_full.explained_variance_ratio_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._pca_reduced.transform(X)

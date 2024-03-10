import numpy as np

from modules.utils.covariance_matrix import covariance_matrix


class MyPCA:
    def __init__(self, n_components):
        """
        Inisialisasi objek MyPCA.

        Parameter:
        - n_components (int): Jumlah komponen utama yang akan dipertahankan.
        """
        self.cum_explained_variance = None
        self.explained_variance_ratio = None
        self.components = None
        self.n_components = n_components

    def fit(self, x):
        """
        Melatih model PCA menggunakan data input.

        Parameter:
        - x (numpy.ndarray): Matriks data input.

        Returns:
        - self: Objek MyPCA yang telah dilatih.
        """
        x = x.copy()

        # Eigendecomposition dari matriks kovarian
        cov_mat = covariance_matrix(x)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        eig_vecs = eig_vecs.T

        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[i, :]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda _x: _x[0], reverse=True)
        eig_vals_sorted = np.array([x[0] for x in eig_pairs])
        eig_vecs_sorted = np.array([x[1] for x in eig_pairs])

        self.components = eig_vecs_sorted[:self.n_components, :]

        # Rasio variansi yang dijelaskan
        self.explained_variance_ratio = [i / np.sum(eig_vals) for i in eig_vals_sorted[:self.n_components]]
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)

        return self

    def transform(self, x):
        """
        Mentransformasi data input menggunakan komponen utama yang telah dilatih.

        Parameter:
        - x (numpy.ndarray): Matriks data input yang akan ditransformasi.

        Returns:
        - transformed_data (numpy.ndarray): Data yang telah ditransformasi menggunakan komponen utama.
        """
        x = x.copy()
        transformed_data = x.dot(self.components.T)
        return transformed_data

    def fit_transform(self, x):
        """
        Melatih model PCA menggunakan data input. Lalu mentransformasi data input menggunakan komponen utama yang telah dilatih.

        Parameter:
        - x (numpy.ndarray): Matriks data input yang akan ditransformasi.

        Returns:
        - transformed_data (numpy.ndarray): Data yang telah ditransformasi menggunakan komponen utama.
        """
        x = x.copy()

        self.fit(x)
        return self.transform(x)
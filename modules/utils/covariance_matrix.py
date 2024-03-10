import numpy as np


def covariance_matrix(x):
    """
    Menghitung matriks kovarian dari data X.

    Parameters:
    - X: np.array, matriks data dengan setiap kolom mewakili suatu fitur.

    Returns:
    - cov_mat: np.array, matriks kovarian dari data X.
    """
    n_samples, n_features = x.shape

    # Menghitung rata-rata setiap fitur
    mean_vector = np.mean(x, axis=0)

    # Menghitung deviasi dari rata-rata
    deviation_matrix = x - mean_vector

    # Menghitung matriks kovarian
    cov_mat = np.dot(deviation_matrix.T, deviation_matrix) / (n_samples - 1)

    return cov_mat

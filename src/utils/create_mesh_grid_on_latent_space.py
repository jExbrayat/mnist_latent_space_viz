from src.models.split_encoder_decoder import split_encoder_decoder
from src.data.import_data import import_data
import numpy as np
import matplotlib.pyplot as plt


def create_meshgrid(encod):
    # Load data to encode
    X, _, _ = import_data()

    # Encod the whole dataset
    X_encod = encod.predict(X)

    X_encod.shape  # X_encod shape is (60_000, 2)

    # Determine lower and upper bounds of the two axes (second dimension of array)
    dim1_lower_bound = np.min(X_encod[:, 0])
    dim2_lower_bound = np.min(X_encod[:, 1])
    dim1_upper_bound = np.max(X_encod[:, 0])
    dim2_upper_bound = np.max(X_encod[:, 1])

    # Make meshgrid on the two dimensional latent space
    n_points_dim1, n_points_dim2 = 100, 100
    dim1_points = np.linspace(dim1_lower_bound, dim1_upper_bound, n_points_dim1)
    dim2_points = np.linspace(dim2_lower_bound, dim2_upper_bound, n_points_dim2)
    grid = np.meshgrid(dim1_points, dim2_points)
    x_coordinates, y_coordinates = grid[0], grid[1]

    return x_coordinates, y_coordinates

from keras.models import load_model
from src.models.split_encoder_decoder import split_encoder_decoder
from src.utils.create_mesh_grid_on_latent_space import create_meshgrid
import numpy as np
import matplotlib.pyplot as plt

# Load complete encoder decoder
model = load_model("models/encod_decod.keras")

# Get decoder
encod, decod = split_encoder_decoder(encoder_decoder_model=model)

# Create meshgrid
x_coordinates, y_coordinates = create_meshgrid(encod=encod)

# Decod each data point in latent space
meshgrid_decod = np.empty(
    shape=(x_coordinates.shape[0], x_coordinates.shape[1], 28, 28)
)  # 100x100 table of images of size 28x28
for row in range(x_coordinates.shape[0]):
    for col in range(x_coordinates.shape[1]):
        # Extract a data point X from grid
        X = np.array([x_coordinates[row, col], y_coordinates[row, col]]).reshape(1, 2)

        # Decod data point
        X_decod = decod.predict(X)

        # Reshape to image size
        X_decod_reshape = np.reshape(X_decod, newshape=(28, 28))

        # Store decoded image into numpy array
        meshgrid_decod[row, col, :, :] = X_decod_reshape

# Save decoded meshgrid into array
np.save("models/decoded_meshgrid/30-01-24_decod_grid", meshgrid_decod)

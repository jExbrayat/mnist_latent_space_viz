from src.utils.create_mesh_grid_on_latent_space import create_meshgrid
from keras.models import load_model
from src.models.split_encoder_decoder import split_encoder_decoder
import matplotlib.pyplot as plt
import numpy as np

# Load model
encod_decod = load_model("models/encod_decod.keras")
classificator = load_model("models/classificator.keras")

# Split encoder and decoder
encod, decod = split_encoder_decoder(encoder_decoder_model=encod_decod)

# Create meshgrid
x_coordinates, y_coordinates = create_meshgrid(encod=encod)


# Predict class for each coordinate
colormap = np.empty(shape=(x_coordinates.shape[0], x_coordinates.shape[1]))
for row in range(x_coordinates.shape[0]):
    for col in range(x_coordinates.shape[1]):
        # Extract a data point X from grid
        X = np.array([x_coordinates[row, col], y_coordinates[row, col]]).reshape(1, 2)

        # Make prediction
        y_pred = classificator.predict(X)

        # Convert back to class indicator
        y_pred_class = np.argmax(y_pred, axis=1)

        # Register prediction into color map
        colormap[row, col] = int(y_pred_class)

# Save colormap
np.save("models/colormaps/30-01-24_cmap", colormap)

# Color grid based on prediction
colormap = colormap.astype(int)
plt.scatter(x_coordinates, y_coordinates, c=colormap)
plt.show()

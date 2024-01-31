import numpy as np
import json

# Load the NumPy array
numpy_array = np.load("models/decoded_meshgrid/30-01-24_decod_grid.npy")

# Convert NumPy array to a regular Python list
python_list = numpy_array.tolist()

# Save the Python list as JSON
with open("apps/your_data.json", "w") as json_file:
    json.dump(python_list, json_file)

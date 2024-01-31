from keras.models import load_model
from src.data.import_data import import_data
import matplotlib.pyplot as plt


# Load model
encod_decod = load_model("models/encod_decod.keras")

# Import data to encode / decode
X, _, _ = import_data()

# Predict i.e. encode and decode image
img = X[[85]]
img_decod = encod_decod.predict(img)

# Reshape image
img_decod = img_decod.reshape(28, 28)
img = img.reshape(28, 28)

# Plot decoded image vs original
fig, axs = plt.subplots(2, 1)
axs[0].imshow(img_decod)
axs[0].set_title("decoded image")
axs[1].imshow(img)
axs[1].set_title("original image")
plt.show()

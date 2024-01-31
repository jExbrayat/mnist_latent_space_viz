from src.data.import_data import import_data
from src.models.define_model import create_encod_decod
from keras.callbacks import EarlyStopping
import src.utils.model_viz as model_viz
from keras.models import save_model

# Import data
X, y, y_categorical = import_data()

# Define encoder decoder model
encod_decod = create_encod_decod()
encod_decod.summary(expand_nested=True)

# Compile
encod_decod.compile(loss="mse", optimizer="adam")

# Define callbacks
stopper = EarlyStopping(monitor="loss", patience=50, restore_best_weights=True)

# Train model
encod_decod_history = encod_decod.fit(X, X, batch_size=32, epochs=200)

# Vizualize model training curves
model_viz.plot_training_curve(
    loss=encod_decod_history.history["loss"],
    plot_from_n_epoch=0,
    save_path="models/training_curves/encod_decod_training.png",
)

# Save encoder decoder
save_model(encod_decod, "models/encod_decod.keras")

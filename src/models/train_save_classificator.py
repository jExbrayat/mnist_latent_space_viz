from src.data.import_data import import_data
from src.models.define_model import create_classificator
from src.models.split_encoder_decoder import split_encoder_decoder
from keras.callbacks import EarlyStopping, History
import src.utils.model_viz as model_viz
from keras.models import save_model

# Import data
X, y, y_categorical = import_data()

# Encode data
encod, _ = split_encoder_decoder()
X_encod = encod.predict(X)
X_encod.shape

# Define classificator
classificator = create_classificator()
classificator.summary()

# Compile classificator
classificator.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics="accuracy"
)

# Define callbacks
stopper = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
classificator_history = History()

# Train model
classificator.fit(
    X_encod,
    y_categorical,
    batch_size=32,
    epochs=300,
    callbacks=[stopper, classificator_history],
    validation_split=0.2,
)

# Vizualize model training curves
model_viz.plot_training_curve(
    loss=classificator_history.history["loss"],
    plot_from_n_epoch=0,
    save_path="models/training_curves/classificator_training.png",
)

# Save encoder decoder
save_model(classificator, "models/classificator.keras")

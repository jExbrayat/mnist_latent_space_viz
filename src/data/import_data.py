from keras.datasets import mnist
from keras.utils import to_categorical


def import_data():
    train, _ = mnist.load_data()
    X, y = train[0], train[1]

    # Preprocess
    X = X.astype("float32") / 255
    X = X.reshape(-1, 784)
    y_categorical = to_categorical(y)
    return X, y, y_categorical

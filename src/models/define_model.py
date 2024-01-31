from keras.layers import Dense
from keras.models import Sequential


def create_encod():
    model = Sequential(
        [
            Dense(128, input_shape=(784,), activation="sigmoid"),
            Dense(32, activation="sigmoid"),
            Dense(8, activation="sigmoid"),
            Dense(2, activation="sigmoid"),
        ]
    )

    return model


def create_decod():
    model = Sequential(
        [
            Dense(8, input_shape=(2,), activation="sigmoid"),
            Dense(32, activation="sigmoid"),
            Dense(128, activation="sigmoid"),
            Dense(784, activation="sigmoid"),
        ]
    )

    return model


def create_encod_decod():
    encod = create_encod()
    decod = create_decod()
    encod_decod = Sequential()
    encod_decod.add(encod)
    encod_decod.add(decod)

    return encod_decod


def create_classificator():
    model = Sequential(
        [
            Dense(8, input_shape=(2,), activation="relu"),
            Dense(16, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    return model

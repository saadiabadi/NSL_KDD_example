import tensorflow
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM



import random



# Create an initial LSTM Model
def create_seed_model():
    model = Sequential()
    model.add(LSTM(100, input_shape=(1,42)))
    model.add(tensorflow.keras.layers.Dense(72, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(50, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(36, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(28, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                  optimizer=tensorflow.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

import tensorflow as tf
from tensorflow import keras
import numpy as np


def make_model(input_shape, output_shape):
    print("Shapes", input_shape, output_shape)
    model = keras.models.Sequential([
        keras.layers.Input(input_shape),
        keras.layers.Conv1D(32, (3, ), padding="same", activation="relu"),
        keras.layers.Conv1D(64, (3, ), padding="same", activation="relu"),
        keras.layers.Conv1D(128, (3, ), padding="same", activation="relu"),
        keras.layers.MaxPool1D((3, ), 2, padding="same"),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(256, "relu"),
        keras.layers.Dense(512, "relu"),
        keras.layers.Dense(output_shape[0], "softmax"),
        keras.layers.Reshape(output_shape)
    ])
    return model

def train(x: np.ndarray, y: np.ndarray):
    # Shuffling data
    idx = np.random.permutation(len(x))
    x = x[idx]
    y = y[idx]

    x_train = x[0: 7*(len(x)//10)]
    x_test = x[7*(len(x)//10): ]
    y_train = y[0: 7*(len(y)//10)]
    y_test = y[7*(len(y)//10): ]

    x_train = x_train.reshape((*x_train.shape, 1))
    x_test = x_test.reshape((*x_test.shape, 1))  
    y_train = y_train.reshape((*y_train.shape, 1))
    y_test = y_test.reshape((*y_test.shape, 1))

    model = make_model(x_train.shape[1:], y_train.shape[1:])
    print(model.summary())
    # keras.utils.plot_model(model, show_shapes=True)

    epochs = 10000
    batch_size = 32

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.00001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy", "categorical_crossentropy"],
    )
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    # test_loss, test_acc = model.evaluate(x_test, y_test)
    # print(test_loss, test_acc)
    print(model.evaluate(x_test, y_test))

    
    model.save("fft_model.h5")

# def test(model):
#     test_loss, test_acc = model.evaluate(x_test, y_test)
from tensorflow import keras
import tensorflow as tf
import numpy as np


def make_model(input_shape):
    model = keras.Sequential()
    model.add(keras.layers.Dense(512, input_shape=input_shape, activation="relu"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1))
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

    # x_train = x_train.reshape((*x_train.shape, 1))
    # x_test = x_test.reshape((*x_test.shape, 1))  
    # y_train = y_train.reshape((*y_train.shape, 1))
    # y_test = y_test.reshape((*y_test.shape, 1))

    model = make_model(x_train.shape[1:])
    print(model.summary())
    # keras.utils.plot_model(model, show_shapes=True)

    epochs = 500
    batch_size = 32

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    model.compile(
        optimizer="SGD",
        loss="mean_squared_error",
        metrics=["cosine_similarity"],
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

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(test_loss, test_acc)
    
    model.save("fft_model_new_1.h5")
    # model.save_weights("fft_model_new_weights.h5")
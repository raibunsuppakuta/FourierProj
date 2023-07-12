import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy import fft
import pandas as pd
import yfinance as yf

from optim import _filter
from augment import normalisation


def main(dat: np.ndarray):
    # Take in data
    if len(dat) != 100:
        raise ValueError(f"Size of input is of length {len(dat)}, incompatible with (100)")
    # fft
    yfft = abs(fft.fftshift(fft.fft(dat)))
    yfft = normalisation(yfft)
    yfft = np.reshape(yfft, (1, -1))
    print(yfft.shape)

    # input fft into model to get cf
    model = keras.models.load_model("fft_model_new_1.h5")
    cf = model.predict(yfft)
    # butterworth yfft with cf
    filtered = _filter(dat, cf)
    return filtered


def plotter(y, y_filtered):
    x = np.arange(0, 100)
    plt.plot(x, y, label="Raw data")
    plt.plot(x, y_filtered, label="Filtered data")
    plt.show()

if __name__ == "__main__":
    dat = np.array(yf.download("AAPL", period="2y")["Close"].iloc[123:223])
    filt = main(dat)
    plotter(dat, filt)



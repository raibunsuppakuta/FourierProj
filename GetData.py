import yfinance as yf
import numpy as np

def get_data(stock):
    df = yf.download(stock, period="max")
    # df["EMA"] = df["Close"].ewm(20).mean()
    df["SMA"] = df["Close"].rolling(20).mean()
    x = np.array(df["SMA"].iloc[20: ])
    rem = len(x) % 100
    x = np.array(np.split(x[rem: ], len(x)//100))
    return x


def corrupt(data: np.ndarray):
    # Applies to entire array
    max = data.max()
    noise_max = 0.1*max
    noisy = data + np.random.uniform(-noise_max, noise_max, data.shape)
    return noisy

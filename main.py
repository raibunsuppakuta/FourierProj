import numpy as np
import scipy as sp
from scipy import fft
import matplotlib.pyplot as plt
import tensorflow as tf

import GetData as gd
import optim as o
import model_new as m
import augment as aug


def main():
    # Get the data
    stock_tickers = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'JPM', 'JNJ', 'V',
    'NVDA', 'PG', 'UNH', 'PYPL', 'MA', 'BAC', 'HD', 'DIS', 'VZ', 'MRK',
    'CMCSA', 'WMT', 'PFE', 'XOM', 'CSCO', 'KO', 'BA', 'NFLX', 'INTC', 'CRM',
    'ABT', 'NKE', 'PEP', 'ADBE', 'T', 'AMD', 'VOD', 'C', 'MCD', 'IBM',
    'QCOM', 'AMGN', 'AVGO', 'BMY', 'CVS', 'PDD', 'COST', 'TSM', 'WFC', 'ACN',
    'NIO', 'ORCL', 'NEE', 'DHR', 'LIN', 'HON', 'LLY', 'NOW', 'ASML', 'SBUX',
    'TMUS', 'ABBV', 'TXN', 'LMT', 'QQQ', 'GM', 'BABA', 'NOK', 'DUK', 'BP',
    'GILD', 'GSK', 'F', 'ING', 'BBVA', 'BKNG', 'PBR', 'LRCX', 'INTU', 'AMD',
    'MU', 'GOOS', 'AAL', 'DELL', 'NKLA', 'GE', 'SPCE', 'NCLH', 'UAL', 'ZM',
    'BYND', 'RKT', 'JMIA', 'RCL', 'CRWD', 'FSLY', 'SNAP', 'ZS', 'SQ', 'PYPL'
]
    print("Getting data")
    pure = gd.get_data(stock_tickers[0])
    for i in stock_tickers[1: ]:
        pure = np.concatenate((pure, gd.get_data(i)), axis=0)
    print("Original number of sets: ", pure.shape)

    original = pure.copy()
    print("Augmentations")
    for i in range(10):
        for f in [aug.add_random, aug.multiply_random, aug.lin_combi]:
            a = np.apply_along_axis(f, 1, original)
            pure = np.concatenate((pure, a), axis=0)
    print("Number of sets: ", pure.shape)
    
    print("Adding noise")
    corrupt = np.apply_along_axis(gd.corrupt, 0, pure)
    
    # curr shape: (x, 100)
    print("fft")
    # ans = np.vectorize(o.optim, signature="(n), (n)->(m)")(pure, corrupt)
    ans = np.vectorize(o.optim, signature="(n), (n)->()")(pure, corrupt)

    # Printing spread of frequencies
    # y = np.sum(ans, 0)
    # freqs = fft.fftshift(fft.fftfreq(int(pure.shape[1])))
    # plt.plot(freqs, y)
    # print(freqs[y!=0])
    # plt.show()

    yfft = np.apply_along_axis(lambda x: abs(fft.fftshift(fft.fft(x))), 0, corrupt)
    yfft = np.apply_along_axis(aug.normalisation, 1, yfft)
    m.train(yfft, ans)


if __name__ == "__main__":
    main()
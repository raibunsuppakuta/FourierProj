import numpy as np
import scipy as sp
from scipy import fft
import matplotlib.pyplot as plt


def _filter(y, cf, order=1):
    nyquist = 0.5
    norm_cf = cf/nyquist
    b, a = sp.signal.butter(order, norm_cf)
    return sp.signal.filtfilt(b, a, y)

def optim(actual: np.ndarray, noisy: np.ndarray):
    def mse(x):
        filtered = _filter(noisy, x)
        return np.mean((filtered - actual)**2)
    
    freqs = fft.fftshift(fft.fftfreq(actual.shape[0]))
    pf = freqs[freqs>0]
    err_func = np.vectorize(mse)
    err_x = err_func(pf)
    min_err = min(err_x)

    # Already one hot encoded
    # return (freqs==pf[err_x==min_err][0]).astype(np.int8)

    return pf[err_x==min_err][0]
    


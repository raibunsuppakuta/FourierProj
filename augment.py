import numpy as np


def add_random(data: np.ndarray):
    augmented = data + np.random.uniform(-max(data), max(data))
    return augmented


def multiply_random(data: np.ndarray):
    augmented = data*np.random.uniform(-2, 2)
    return augmented


def lin_combi(data: np.ndarray):
    augmented = (data + np.random.uniform(-max(data), max(data)))*np.random.uniform(-2, 2)
    return augmented


def normalisation(arr: np.ndarray):
    max_val = np.max(arr)
    min_val = np.min(arr)
    return (arr - min_val) / (max_val - min_val)
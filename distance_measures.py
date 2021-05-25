import numpy as np


def KLD(a, b):
    pass


def Fidelity(a: np.array, b: np.array):
    return np.abs(np.sum(np.conj(a) * b, axis=-1, keepdims=True))


DISTANCE_MEASURES = {
    'KLD': KLD,
    "Fidelity": Fidelity,
}
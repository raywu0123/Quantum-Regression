from functools import reduce
from typing import Tuple

import numpy as np
from scipy.stats import unitary_group

from distance_measures import DISTANCE_MEASURES


def kronecker(A, B):
    return np.einsum("ab,cd->acbd", A, B).reshape(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])


def get_product(n_qubit: int, num_data: int) -> Tuple[np.array, np.array]:
    data = []
    qubit_data = []
    for _ in range(num_data):
        qubits = []
        for _ in range(n_qubit):
            state = np.zeros([1, 2])
            state[:, 0] = 1
            random_unitary = unitary_group.rvs(2)
            state = state @ random_unitary
            qubits.append(state)
        
        data.append(reduce(lambda x, y: kronecker(x, y), qubits))
        qubit_data.append(np.concatenate(qubits))

    return np.concatenate(data), np.asarray(qubit_data)


def get_entangled(n_qubit: int, num_data: int) -> np.array:
    data = []
    for _ in range(num_data):
        state = np.zeros([1, 2 ** n_qubit])
        state[:, 0] = 1
        random_unitary = unitary_group.rvs(2 ** n_qubit)
        state = state @ random_unitary
        data.append(state)

    return np.concatenate(data)


class DistanceDataset:

    def __init__(self, n_qubit: int, distance_measure: str) -> None:
        self.n_qubit = n_qubit
        self.distance_measure_fn = DISTANCE_MEASURES[distance_measure]

    def get_training_data(self, num_data: int) -> Tuple[np.array, np.array, np.array]:
        s_1, _ = get_product(self.n_qubit, num_data)
        s_2, _ = get_product(self.n_qubit, num_data)
        return s_1, s_2, self.distance_measure_fn(s_1, s_2)

    def get_testing_data(self, num_data: int) -> Tuple[np.array, np.array, np.array]:
        s_1 = get_entangled(self.n_qubit, num_data)
        s_2 = get_entangled(self.n_qubit, num_data)
        return s_1, s_2, self.distance_measure_fn(s_1, s_2)

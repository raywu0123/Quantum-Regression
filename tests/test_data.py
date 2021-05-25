from data import get_entangled, get_product, DistanceDataset
import numpy as np

NUM_DATA = 10
N_QUBIT = 3


def test_product():
    data, qubit_data = get_product(N_QUBIT, NUM_DATA)
    
    assert tuple(data.shape) == (NUM_DATA, 2 ** N_QUBIT)
    assert tuple(qubit_data.shape) == (NUM_DATA, N_QUBIT, 2)
    np.testing.assert_array_almost_equal(
        np.linalg.norm(data, axis=1),
        1.
    )


def test_entangled():
    data = get_entangled(N_QUBIT, NUM_DATA)
    assert tuple(data.shape) == (NUM_DATA, 2 ** N_QUBIT)

    np.testing.assert_array_almost_equal(
        np.linalg.norm(data, axis=1),
        1.
    )


def test_training_data():
    dataset = DistanceDataset(n_qubit=N_QUBIT, distance_measure='Fidelity')
    s1, s2, f = dataset.get_training_data(NUM_DATA)
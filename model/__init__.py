from .nn import NeuralNetworkModel
from .vqc import VariationalQuantumCircuitModel

MODEL_HUB = {
    'NN': NeuralNetworkModel,
    'VQC': VariationalQuantumCircuitModel,
}
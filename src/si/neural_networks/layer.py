import numpy as np


class Dense:
    """
    A dense layer is a layer where each neuron is connected to all neurons in the previous layer.

    Parameters
    ----------
    input_size: int
        The number of inputs the layer will receive.
    output_size: int
        The number of outputs the layer will produce.

    Attributes
    ----------
    weights: np.ndarray
        The weights of the layer.
    bias: np.ndarray
        The bias of the layer.
    X : np.ndarray
        The input to the layer.
    """
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the dense layer.

        Parameters
        ----------
        input_size: int
            The number of inputs the layer will receive.
        output_size: int
            The number of outputs the layer will produce.
        """
        # parameters
        self.input_size = input_size
        self.output_size = output_size

        # attributes
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.X = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).

        Parameters
        ----------
        input: np.ndarray
            The input to the layer.
        """
        self.X = input
        return np.dot(input, self.weights) + self.bias


class SigmoidActivation:
    """
    A sigmoid activation layer.

    Attributes
    ----------
    X : np.ndarray
        The input to the layer.
    """
    def __init__(self):
        """
        Initialize the sigmoid activation layer.
        """
        self.X = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).

        Parameters
        ----------
        input: np.ndarray
            The input to the layer.
        """
        self.X = input
        return 1 / (1 + np.exp(-input))

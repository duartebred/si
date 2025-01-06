from abc import abstractmethod
from typing import Union

import numpy as np

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Base class for activation layers.
    """

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error: float) -> Union[float, np.ndarray]:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output error of the layer.
        """
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Derivative of the activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The derivative of the activation function.
        """
        raise NotImplementedError

    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        tuple
            The output shape of the layer.
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0
    
class SigmoidActivation(ActivationLayer):
    """
    Sigmoid activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return 1 / (1 + np.exp(-input))

    def derivative(self, input: np.ndarray):
        """
        Derivative of the sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input) * (1 - self.activation_function(input))


class ReLUActivation(ActivationLayer):
    """
    ReLU activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return np.maximum(0, input)

    def derivative(self, input: np.ndarray):
        """
        Derivative of the ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return np.where(input >= 0, 1, 0)

class TanhActivation(ActivationLayer):
    """
    TanhActivation applies the hyperbolic tangent function element-wise, 
    compressing input values to a range between -1 and 1.
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Compute the hyperbolic tangent of the input.

        Parameters
        ----------
        input : np.ndarray
            Input array to which the tanh function will be applied.

        Returns
        -------
        np.ndarray
            Output array with the tanh function applied element-wise.
        """
        return np.tanh(input)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the tanh function.

        Parameters
        ----------
        input : np.ndarray
            Input array for which the derivative will be calculated.

        Returns
        -------
        np.ndarray
            Output array with the derivative of tanh applied element-wise.
        """
        tanh_output = self.activation_function(input)
        return 1 - tanh_output ** 2

class SoftmaxActivation(ActivationLayer):
    """
    SoftmaxActivation converts raw output scores into probabilities, 
    making it suitable for multi-class classification tasks.
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Apply the softmax function to the input to compute probabilities.

        Parameters
        ----------
        input : np.ndarray
            Input array where softmax will be applied.

        Returns
        -------
        np.ndarray
            Output array with probabilities, summing to 1 for each sample.
        """
        # Subtract the maximum for numerical stability
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the softmax function.

        Parameters
        ----------
        input : np.ndarray
            Input array for which the derivative will be computed.

        Returns
        -------
        np.ndarray
            Output array with the derivative of the softmax applied element-wise.
        """
        softmax_output = self.activation_function(input)
        return softmax_output * (1 - softmax_output)

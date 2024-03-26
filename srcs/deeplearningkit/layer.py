from abc import ABC, abstractmethod
from .initializer import Initializer
import numpy as np

class Layer(ABC):
	output: np.ndarray
	input: np.ndarray
	weights: np.ndarray
	biases: np.ndarray
	dweigths: np.ndarray
	dinputs: np.ndarray
	dbiases: np.ndarray

	def __init__(self, n_inputs, n_neurons, initializer: Initializer=None, weight_regularizer_l1 = 0, bias_regularizer_l1 = 0, weight_regularizer_l2 = 0, bias_regularizer_l2 = 0):
		self.n_inputs = n_inputs
		self.n_neurons = n_neurons
		self.shape = (n_inputs, n_neurons)
		self.initializer = initializer
		self.initialize_weights()
		self.weight_regularizer_l1 = weight_regularizer_l1
		self.weight_regularizer_l2 = weight_regularizer_l2
		self.bias_regularizer_l1 = bias_regularizer_l1
		self.bias_regularizer_l2 = bias_regularizer_l2

	def initialize_weights(self):
		if self.initializer is not None:
			self.weights = self.initializer(self.shape)
		else:
			self.weights = 0.10 * np.random.randn(self.n_inputs, self.n_neurons)
		self.biases = np.zeros((1, self.n_neurons))

	def updateNeurons(self, n_neurons):
		self.n_neurons = n_neurons
		self.shape = (self.n_inputs, n_neurons)
		self.initialize_weights()

	def updateInputs(self, n_inputs):
		self.n_inputs = n_inputs
		self.shape = (n_inputs, self.n_neurons)
		self.initialize_weights()

	@abstractmethod
	def forward(self, inputs):
		pass
	@abstractmethod
	def backward(self, dvalues):
		pass

class Dense(Layer):
	def __init__(self, n_inputs, n_neurons, initializer: Initializer = None):
		super().__init__(n_inputs, n_neurons, initializer)
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases
		self.input = inputs
	def backward(self, dvalues):
		self.dweights = np.dot(self.input.T, dvalues)
		self.dinputs = np.dot(dvalues, self.weights.T)
		self.dbiases = np.sum(dvalues, axis=0, keepdims=True)


def layer(layer_type: str, *args) -> Layer:
	layer_type = layer_type.lower()
	if layer_type == "dense":
		return Dense(*args)
	else:
		raise ValueError(f"LayerError: Unknown layer type: {layer_type}")

__all__ = ['Layer', 'Dense', 'layer']
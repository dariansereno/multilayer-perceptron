import numpy as np

class Initializer:
	def __call__(self, shape: tuple) -> np.ndarray:
		pass

# Distribution normal pour tout les neuronnes
class RandomNormal(Initializer):
	def __init__(self, mean=0.5, std=1, **kwargs):
		self.mean = mean
		self.std = std
	def __call__(self, shape: tuple) -> np.ndarray:
		return np.random.normal(self.mean, self.std, shape)
# Zero a tout les neuronnes
class Zero(Initializer):
	def __init__(self):
		pass

	def __call__(self, shape: tuple) -> np.ndarray:
		return np.zeros(shape)

#Distribution normal entre l'écart-type 2 / n_inputs, fait pour eviter le vanishing gradient avec ReLU
class HeNormal(Initializer):
	def __init__(self, n_inputs=10, **kwargs):
		self.n_inputs = n_inputs

	def __call__(self, shape: tuple) -> np.ndarray:
		std = np.sqrt(2.0 / self.n_inputs)
		return np.random.randn(*shape) * std

# Distribution uniforme entre l'ecart-type -6 / n_input => 6 / n_inputs.
# Variante de HeNormal en uniforme
class HeUniform(Initializer):
	def __init__(self, n_inputs=2, **kwargs):
		self.n_inputs = n_inputs

	def __call__(self, shape: tuple) -> np.ndarray:
		std = np.sqrt(6.0 / self.n_inputs)
		return np.random.uniform(-std, std, shape)

#Distrubition uniforme entre l'intervalle  -6 / (n_input - n_outputs) =>  6 / (n_input - n_outputs)
#Evite le vanishing et exploding gradient
class Xavier(Initializer):
	def __init__(self, n_inputs=2, n_outputs=2, *kwargs):
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs

	def __call__(self, shape: tuple) -> np.ndarray:
		limit = np.sqrt(6 / (self.n_inputs + self.n_outputs))
		return np.random.uniform(-limit, limit, shape)

# Distribution uniforme entre - 1/ sqrt(n_inputs) => 1 / sqrt(n__inputs)
# Pour Sigmoid et TanH
class LeCun(Initializer):
	def __init__(self, n_inputs, **kwargs):
		self.n_inputs = n_inputs
	
	def __call__(self, shape: tuple) -> np.ndarray:
		limit = np.sqrt(1 / self.n_inputs)
		return np.random.uniform(-limit, limit, shape)

def initializer(initializer_type: str, *args, **kwargs) -> Initializer:
	initializer_type = initializer_type.lower()
	if args and isinstance(args[-1], dict):
		kwargs.update(args[-1])
		args = args[:-1]

	if (initializer_type == "randomnormal"):
		return RandomNormal(*args, **kwargs)
	if (initializer_type == "zero"):
		return Zero()
	if (initializer_type == "henormal"):
		return HeNormal(*args, **kwargs)
	if (initializer_type == "heuniform"):
		return HeUniform(*args, **kwargs)
	if (initializer_type == "xavier"):
		return Xavier(*args, **kwargs)
	if (initializer_type == "lecun"):
		return LeCun(*args, **kwargs)
	raise ValueError(f"InitializerError: Unknown initializer type :'{initializer_type}'")

__all__ = ['Initializer', 'RandomNormal', 'Zero', 'HeNormal', "HeUniform", 'Xavier', 'Lecun', 'initializer']
from abc import ABC, abstractmethod
import numpy as np
from .layer import Layer

# Ces optimizers utilise pour la plupart l'Exponentially Weighted Moving Average (EWMA), qui permet
# un descente plus smooth et plus calibrable.
# Par exemple, pour une descente avec momemtum l'EWMA = (βVt + (1-β)ΔWt).
# Ou β est le momentum [0, 1], Vt la vélocité actuelle ((βVt + (1-β)ΔWt) fait au tour d'avant, ou 0 si premier tour)
# et ΔWt le gradient des poids de la couche. (Calcul equivalent pour les biais : (βVt + (1-β)ΔBt))
#
# Le β permet de specifie ce qui a le + d'importance entre la partie droite et gauche du calcul, ici
# le momemtum de la descente (Vt) ou la valeur actuelle (ΔWt).
# Si β = 0.5 : Vt * 0.5 ET (1-0.5)ΔWt = 0.5 * ΔWt. Les 2 ont le meme poids
# Si β = 0.7 : Vt * 0.7 ET (1-0.7)ΔWt = 0.3 * ΔWt. La velocite a plus de poids que le gradient.
# Si β = 0.3 : Vt * 0.3 ET (1-0.3)ΔWt = 0.7 * ΔWt. Le gradient a plus de poids que la velocite

class Optimizer(ABC):
	def __init__(self, learning_rate=1.0, decay=.0):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
	@abstractmethod
	def pre_update_params(self):
		pass

	@abstractmethod
	def update_params(self, layer: Layer):
		pass

	@abstractmethod
	def post_update_params(self):
		pass

class SGD(Optimizer):
	def __init__(self, learning_rate=1.0, decay=0.0, momentum=.0, **kwargs):
		super().__init__(learning_rate, decay)
		self.momentum = momentum

	def pre_update_params(self):
		if (self.decay):
			self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

	# From my test, option 2 and 1 are pretty similar. But Option 1 use the Exponentially Weighted Moving Average (EWMA), more lisible, more comprehensive that
	# the option 2, and normally is the most optimized way. See top of the page.
	
	def update_params(self, layer):
		if (self.momentum):
			if not hasattr(layer, 'cached_weights'):
				layer.cached_weights = np.zeros_like(layer.weights)
				layer.cached_biases = np.zeros_like(layer.biases)
			weights_update = self.momentum * layer.cached_weights + (1 - self.momentum) * layer.dweights # OPTION 1
			#weights_update = self.momentum * layer.cached_weights - self.current_learning_rate * layer.dweights # OPTION 2
			layer.cached_weights = weights_update

			#biases_update = self.momentum * layer.cached_biases - self.current_learning_rate * layer.dbiases  # OPTION 2
			biases_update = self.momentum * layer.cached_biases + (1 - self.momentum) * layer.dbiases # OPTION 1
			layer.cached_biases = biases_update
		else:
			weights_update = -self.current_learning_rate * layer.dweights
			biases_update = -self.current_learning_rate * layer.dbiases
		
		#layer.weights += weights_update  # OPTION 2
		#layer.biases += biases_update # OPTION 2
		layer.weights -= weights_update * self.current_learning_rate # OPTION 1
		layer.biases -= biases_update * self.current_learning_rate # OPTION 1

	
	def post_update_params(self):
		self.iterations += 1

class Adagrad(Optimizer):
	def __init__(self, learning_rate=1.0, decay=.0, epsilon=1e-7, **kwargs):
		super().__init__(learning_rate, decay)
		self.epsilon = epsilon

	def pre_update_params(self):
		if (self.decay):
			self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))
	
	def update_params(self, layer: Layer):
		if not hasattr(layer, "cached_weights"):
			layer.cached_weights = np.zeros_like(layer.weights)
			layer.cached_biases = np.zeros_like(layer.biases)

		layer.cached_weights += layer.dweights ** 2
		layer.cached_biases += layer.dbiases ** 2
		layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.cached_weights) + self.epsilon)
		layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.cached_biases) + self.epsilon)

	def post_update_params(self):
		self.iterations += 1

class RMSProp(Optimizer):
	def __init__(self, learning_rate=0.001, decay=.0, epsilon=1e-7, rho=0.9, **kwargs):
		super().__init__(learning_rate, decay)
		self.epsilon = epsilon
		self.rho = rho


	def pre_update_params(self):
		if (self.decay):
			self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))
	
	def update_params(self, layer: Layer):
		if not hasattr(layer, "cached_weights"):
			layer.cached_weights = np.zeros_like(layer.weights)
			layer.cached_biases = np.zeros_like(layer.biases)
		
		layer.cached_weights = self.rho * layer.cached_weights + (1 - self.rho) * (layer.dweights ** 2) 
		layer.cached_biases = self.rho * layer.cached_biases + (1 - self.rho) * (layer.dbiases ** 2)
		# rho method is a EWMA with Velocity on left and gradient^2 on right

		layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.cached_weights) + self.epsilon)
		layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.cached_biases) + self.epsilon)
		# here, we update the weight in terms of the product of learning_rate and dweights divided by the squareroot of the cache. A lil obscure for me rn
		# but I think is to avoid dividing by enormous value sqrt(1000000) = 1000. More the precedent velocity was important, 
		# more the the division will be big, and littler will be the update (avoid too much momemtum certainly, visually,
		# the gradient descent will avoid zig zaging and will be more straight forward to the minima) 
		# Epsilon avoid dividing by zero.

	def post_update_params(self):
		self.iterations += 1

class Adam(Optimizer):
	def __init__(self, learning_rate=0.001, decay=.0, epsilon=1e-7, beta1=0.9, beta2=0.999, **kwargs):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.epsilon = epsilon
		self.beta1 = beta1
		self.beta2 = beta2
		self.iterations = 0

	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

	def update_params(self, layer: Layer):
		if not hasattr(layer, "cached_weights"):
			layer.cached_weights = np.zeros_like(layer.weights)
			layer.cached_biases = np.zeros_like(layer.biases)
			layer.weights_momentum = np.zeros_like(layer.weights)
			layer.biases_momentum = np.zeros_like(layer.biases)

		#beta1 regulate the impact of momentum and dweights. more beta1 is near from 1 and more momentum will have impact,
		# and more beta1 near from 0 gradients (dweights) will have impact
		layer.weights_momentum = self.beta1 * layer.weights_momentum + (1 - self.beta1) * layer.dweights
		layer.biases_momentum = self.beta1 * layer.biases_momentum + (1 - self.beta1) * layer.dbiases

		# wm / (1 - beta1^(iter+1)) -> more we advance, bigger from 1 the number will be. So more the training advance, less the result will be.
		# example : weights_momentum = 5, beta1 = 0.9, iteration = 0
		#		  : 5 / (1 - 0.9^1) = 5 / (1 - 0.9) = 5 / 0.1 = 50
		# iteration = 100
		#		  : 5 / (1 - 0.9^100) ≈ 5 / (1 - 0,00002) ≈ 5 / 0,99998 ≈ 5,0001
		# another example : 5 / (1 - 0.999999^1) = 5 000 000,
		#				  : 5 / (1 - 0.999999^100) ≈ 5 002,50
		# SO, more the beta is near of 1, more the weigths_momentum will start HIGH. If beta is near from zero the, the momentum will be nearly equal to the actual momentum
		# And more the iteration grow up, more the the beta impact (not exactly true but i think you understand me) will decrease
		weights_momentum_corrected = layer.weights_momentum / (1 - self.beta1 ** (self.iterations + 1)) # we need to start at exponent 1
		biases_momentum_corrected = layer.biases_momentum / (1 - self.beta1 ** (self.iterations + 1))

		layer.cached_weights = self.beta2 * layer.cached_weights + (1 - self.beta2) * layer.dweights ** 2 
		layer.cached_biases = self.beta2 * layer.cached_biases + (1 - self.beta2) * layer.dbiases ** 2
		# Î_ more beta2 is near from 1, and less the impact of incoming
		#weights gradient will have an impact, and more the cached weights will have an impact

		cached_weights_corrected = layer.cached_weights / (1 - self.beta2 **(self.iterations + 1))
		cached_biases_corrected = layer.cached_biases / (1 - self.beta2 **(self.iterations + 1))
		# same logic as above, more beta2 is high, more cached_weights will have an impact ad beginning, and will decrease at each iteration.

		layer.weights += -self.current_learning_rate * weights_momentum_corrected / (np.sqrt(cached_weights_corrected) + self.epsilon)
		layer.biases += -self.current_learning_rate * biases_momentum_corrected / (np.sqrt(cached_biases_corrected) + self.epsilon)
		# epsilon is a constant to avoid division by zero

	def post_update_params(self):
		self.iterations += 1

def optimizer(optimizer:str,*args, **kwargs)->Optimizer:
	optimizer = optimizer.lower()
	if args and isinstance(args[-1], dict):
		kwargs.update(args[-1])
		args = args[:-1] 

	if (optimizer == "sgd"):
		return SGD(**kwargs)
	if (optimizer == "adagrad"):
		return Adagrad(**kwargs)
	if (optimizer == "rmsprop"):
		return RMSProp(**kwargs)
	if (optimizer == "adam"):
		return Adam(**kwargs)
	return None

__all__ = ['Optimizer', 'SGD', 'Adagrad', 'RMSProp', 'Adam', 'optimizer']
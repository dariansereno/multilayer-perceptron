from . import layer as Layer
from . import activation as Activation
from . import loss as Loss
import numpy as np
import matplotlib.pyplot as plt
from . import EarlyStopping

class Model:
	def __init__(self):
		self.n_layer = 0
		self.layers = []
		self.activations = []
		self.accuracies = []
		self.losses = []
		self.elapsed_epochs = 0
		self.early_stopping: EarlyStopping.EarlyStopping = None

	def compile(self, optimizer=None, loss=None, early_stopping=None):
		self.optimizer = optimizer
		self.loss = loss
		self.early_stopping = early_stopping

	def add(self, layer: Layer.Layer, activation: Activation):
		self.n_layer += 1
		self.layers.append(layer)
		self.activations.append(activation)
	
	def predict(self, values):
		if isinstance(self.loss, Loss.BinaryCrossEntropy):
			if values.ndim == 1 or values.shape[1] == 1:  # Cas binaire avec une seule sortie
        # Convertir en prédictions binaires (0 ou 1) avec un seuil de 0.5
				predictions = (values > 0.5) * 1
    # Si les prédictions sont issues d'une activation softmax (multi-classes, one-hot)
			elif values.ndim > 1:  # Cas multi-classes ou one-hot encoded
				# Trouver l'indice de la classe avec la plus grande probabilité
				predictions = np.argmax(values, axis=1)
		if isinstance(self.loss, Loss.CategoricalCrossEntropy):
			predictions = np.argmax(values, axis=1)
		return predictions
	
	def fit(self, x=None, y=None, batch_size=None, epochs=1, shuffle=True, display=False, plot=False):
		self.accuracies.clear()
		self.losses.clear()
		loss_activation = None
		x = np.array(x)
		y = np.array(y)
		self.val_accuracies = [] 
		self.val_losses = []    
		steps = 1

		if (batch_size):
			steps = len(x) // batch_size
			if (steps * batch_size < len(x)):
				steps += 1

		for epoch in range(epochs):
			if shuffle:
				indices = np.random.permutation(len(x))
				x = x[indices]
				y = y[indices]
			layer: Layer
			activation: Activation
			epoch_accuracies = []
			epoch_losses = []
			
			for step in range(steps):
				if (batch_size):
					batch_X = x[step*batch_size : (step+1) *batch_size]
					batch_Y = y[step*batch_size : (step+1) *batch_size]
				else:
					batch_X = x
					batch_Y = y
				feed = batch_X

				# forward
				for (layer, activation) in zip(self.layers, self.activations):
					layer.forward(feed)
					feed = layer.output


					# If we have a Loss Activation, it take one more input. So we have to differenciate them and we get the loss here
					if isinstance(activation, Activation.Softmax_CategoricalCrossEntropy):
						loss = activation.forward(feed, batch_Y)
						loss_activation = activation;
					else:
						activation.forward(feed)
					feed = activation.output

				predictions = self.predict(feed)
				if (batch_Y.ndim == 1 or( batch_Y.ndim == 2 and batch_Y.shape[1] == 1)):
					accuracy = np.mean(predictions==batch_Y)
				else:
					labels = np.argmax(batch_Y, axis=1)
					accuracy = np.mean(predictions==labels)
				epoch_accuracies.append(accuracy)

				if loss_activation == None:
					loss = self.loss.calculate(feed, batch_Y)
					self.loss.backward(feed, batch_Y)
					feed = self.loss.dinputs
				# If we have a Loss Activation, we have to do the backward pass directly on the activtation function, the loss is already calculated
				else:
					activation.backward(feed, batch_Y)
					feed = activation.dinputs

				epoch_losses.append(loss)
				# backward
				for (layer, activation) in zip(reversed(self.layers), reversed(self.activations)):
					# since we have already compute the backward for the Loss and the activation, we do not have to compute it here
					if (isinstance(activation, Activation.Softmax_CategoricalCrossEntropy)):
						pass
					else:
						activation.backward(feed)
						feed = activation.dinputs
					layer.backward(feed)
					feed = layer.dinputs
			metrics = self.evaluate(x, y)
			if self.early_stopping and self.early_stopping.should_stop(metrics):
				break
			if (display):
				print(f'epoch {epoch}/{epochs} - loss: {loss:.3f} - val_loss: {metrics["val_loss"]:.3f}')
				
				# update
				self.optimizer.pre_update_params()
				for layer in self.layers:
					self.optimizer.update_params(layer)
				self.optimizer.post_update_params()
			self.elapsed_epochs += 1
			self.val_accuracies.append(metrics["val_accuracy"])
			self.val_losses.append(metrics["val_loss"])
			avg_accuracy = np.mean(epoch_accuracies)
			avg_loss = np.mean(epoch_losses)
			self.accuracies.append(avg_accuracy)
			self.losses.append(avg_loss)

		if (plot):
			self.plot(self.elapsed_epochs)

	def evaluate(self, x, y):
		x = np.array(x)
		y = np.array(y)
		feed = x
		loss_activation = None

		layer: Layer.Layer
		activation: Activation
		for (layer, activation) in zip(self.layers, self.activations):
			layer.forward(feed)
			feed = layer.output
			if (isinstance(activation, Activation.Softmax_CategoricalCrossEntropy)):
				loss = activation.forward(feed, y)
				loss_activation = activation;
			else:
				activation.forward(feed)
			feed = activation.output
		if loss_activation == None:
			loss = self.loss.calculate(feed, y)
		else:
			pass
		predictions = self.predict(feed)
		if (y.ndim == 1 or( y.ndim == 2 and y.shape[1] == 1)):
			accuracy = np.mean(predictions==y)
			true_table = [1 if pred == y_true else 0 for pred, y_true in zip(predictions, y)]
		else:
			labels = np.argmax(y, axis=1)
			accuracy = np.mean(predictions==labels)
			true_table = [1 if pred == y_true else 0 for pred, y_true in zip(predictions, labels)]

		

		return {"prediction": predictions, "val_accuracy": accuracy, "val_loss": loss,"true_table" : true_table}
		#return {"predictions": predictions, "accuracy": accuracy, "loss": loss, "true_table": true_table}
	
	def plot(self, epochs):
		plt.figure(figsize=(12, 6))

		# loss
		plt.subplot(1, 2, 1)
		print(self.val_losses.__len__())
		print(self.val_accuracies.__len__())
		plt.plot(range(epochs), self.losses, label='Training Loss', color='blue')
		if hasattr(self, 'val_losses') and self.val_losses:  # Vérifie si val_losses existe
				plt.plot(range(epochs), self.val_losses, label='Validation Loss', color='red')
		plt.title('Loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()

		# accuracy
		plt.subplot(1, 2, 2)
		plt.plot(range(epochs), self.accuracies, label='Training Accuracy', color='orange')
		if hasattr(self, 'val_accuracies') and self.val_accuracies:  # Vérifie si val_accuracies existe
				plt.plot(range(epochs), self.val_accuracies, label='Validation Accuracy', color='green')
		plt.title('Accuracy')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.legend()

		plt.tight_layout()
		plt.show()

__all__ = ['Model']
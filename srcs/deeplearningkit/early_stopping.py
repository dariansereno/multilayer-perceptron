from enum import Enum
from typing import Literal

class EarlyStopping:
	monitor:Literal["val_accuracy", "val_loss"]
	min_delta:float
	patience:int
	mode: Literal["min", "max"]
	counter: int
	best_metric: float

	def __init__(self, monitor="val_loss", min_delta=0, patience=0,  **kwargs):
		self.monitor = monitor
		self.min_delta = min_delta
		self.patience = patience
		self.mode = "min"
		if (monitor=="val_accuracy"):
			self.mode = "max" 
		self.best_metric = None
		self.counter = 0  # Nombre d'epochs sans amélioration significative

	def should_stop(self, logs):	
		if self.monitor not in logs:
			raise KeyError(f"The monitored metric '{self.monitor}' is not found in logs.")

		if (self.best_metric == None):
			self.best_metric = logs[self.monitor]
			return False
		
		if (self.mode == "min"):
			delta = abs(self.best_metric - logs[self.monitor])
		elif (self.mode == "max"):
			delta = abs(logs[self.monitor] - self.best_metric)
		else:
			raise ValueError("mode should be 'min' or 'max'")
		if (delta > self.min_delta):
			self.best_metric = logs[self.monitor]
			self.counter = 0
			return False
		else:
			self.counter += 1
		
		if (self.counter >= self.patience):
			return True
		
		return False

def early_stopping(*args, **kwargs) -> EarlyStopping:
	if args and isinstance(args[-1], dict):
		kwargs.update(args[-1])
	return EarlyStopping(**kwargs)
	
	
__all__ = ["EarlyStoping"]
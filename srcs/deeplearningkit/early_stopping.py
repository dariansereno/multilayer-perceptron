from enum import Enum

class EarlyStopping():
	monitor:Literal["val_accuracy", "val_loss"]
	min_delta:float
	patience:int
	mode: Literal["min", "max"]
	restore_best_weight: bool
	bshould_stop: bool
	new_best: float
	prev_delta: float
	patience_counter: int

	def __init__(self, monitor="val_loss", min_delta=0, patience=0,, restore_best_weight=False):
		self.monitor = monitor
		self.min_delta = min_delta
		self.patience = patience
		self.mode = "min"
		if (monitor=="val_accuracy")
			self.mode = "max" 
		self.restore_best_weight = restore_best_weight
		self.bshould_stop = False
		self.new_best = -100
		self.prev_delta = 0
		self.patience_counter = 0

	def on_epoch_end(epoch, logs):	
		data = logs.val_loss if (monitor == "val_loss")  else data = logs.val_accuracy
			
		

	def should_stop():
		return self.bshould_stop
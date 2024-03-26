# DeepLearningKit

**Description:**

Welcome to DeepLearningKit, a Python library crafted with a passion for learning and a drive to simplify the journey into the depths of neural networks. As my inaugural project in this domain, DeepLearningKit is designed primarily for my educational purposes, aiming to help me to understand and experiment with deep learning concepts.
This release marks the first version of DeepLearningKit, but the journey doesn't end here! I'll be actively working on improving and expanding the library regularly. Stay tuned for updates and new features in future releases.

**Getting Started:**

1. **Usage:**
   ```python # Import necessary libraries
	import deeplearningkit as nn
	import numpy as np
	from preprocess import preprocess_binary_output_data, create_data

	# Load and create synthetic data (not in the kit)
	(train_X, train_Y), (test_X, test_Y) = create_data(100, 2)

	# Define and compile the neural network model
	model = nn.Model()
	model.add(nn.Layer.Dense(train_X.shape[1], 24, nn.Initializer.HeUniform(train_X.shape[1])), nn.Activation.ReLU())
	model.add(nn.Layer.Dense(24, 24, nn.Initializer.HeUniform(24)), nn.Activation.ReLU())
	model.add(nn.Layer.Dense(24, 2, nn.Initializer.HeUniform(24)), nn.Activation.Softmax())
	model.compile(nn.Optimizer.Adam(), nn.Loss.CategoricalCrossEntropy())

	# Train the model
	model.fit(x=train_X, y=train_Y, batch_size=None, epochs=2001, shuffle=True, display=True, plot=False)

	# Evaluate the trained model
	model.evaluate(test_X, test_Y)

	# Alternatively, load data directly from a CSV file using a preprocessing function (not yet in the kit)
	(train_X, test_X), (train_Y, test_Y) = preprocess_binary_output_data("data.csv")

	# Define and compile the model using a simpler function call
	model = nn.Model()
	model.add(nn.Layer.layer("dense", train_X.shape[1], 24, nn.Initializer.initializer("Heuniform", train_X.shape[1])), nn.Activation.activation('relu'))
	model.add(nn.Layer.layer("dense", 24, 24, nn.Initializer.initializer("heuniform", 24)), nn.Activation.activation("ReLU"))
	model.add(nn.Layer.layer("dense", 24, 1, nn.Initializer.initializer("henormal", 24)), nn.Activation.activation("Sigmoid"))
	model.compile(nn.Optimizer.optimizer("adam"), nn.Loss.loss("BinaryCrossEntropy"))

	# Train the model
	model.fit(x=train_X, y=train_Y, batch_size=None, epochs=1001, shuffle=True, display=True, plot=False)

	# Evaluate the model
	model.evaluate(test_X, test_Y)

	# Load a model configuration from a JSON file
	model_data = nn.parse_model_json("model.json")

	# Compile and fit the model using parsed data and a specified data preprocessing function
	model: nn.Model = nn.compile_and_fit_parsed_model(model_data, preprocess_binary_output_data, display=True, plot=False)
	model.evaluate(test_X, test_Y)

	# Or load, compile, fit, and evaluate directly from a JSON configuration
	model_data = nn.parse_model_json("model.json")
	model: nn.Model = nn.compile_fit_evaluate_parsed_model(model_data, preprocess_binary_output_data, display=True, plot=False)

	# Example of JSON configuration for a neural network model
	# {
	#     "loss_function": "BinaryCrossEntropy",
	#     "layers": [
	#         {"type": "Dense", "units": 24, "activation": "ReLU", "weights_initializer": {"type": "heUniform", "params": {"n_inputs": 2}}},
	#         {"type": "Dense", "units": 24, "activation": "ReLU", "weights_initializer": {"type": "heUniform", "params": {"n_inputs": 24}}},
	#         {"type": "Dense", "units": 1, "activation": "Sigmoid", "weights_initializer": {"type": "xavier", "params": {"n_inputs": 24, "n_outputs": 1}}}
	#     ],
	#     "epochs": 1001,
	#     "batch_size": 0,
	#     "optimizer": {"type": "Adam", "params": {}},
	#     "data": "data.csv"
	# }
   ```

**License:**
DeepLearningKit is licensed under the [MIT License](LICENCE).

**Acknowledgments:**
This project wouldn't have been possible without the support and guidance from the deep learning community, bigup to the book : 'Neural Network From Scratch' by Harrison Kinsley and Daniel Kukiela <3 . Let's continue to learn!

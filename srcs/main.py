import deeplearningkit as nn

import numpy as np
from preprocess import preprocess_binary_output_data
from spiral_data import create_data
from tools import extract_csv
import pandas as pd

if __name__ == "__main__":
	#Model: nn.Model = nn.parse_model_json("../model.json")
	dataset: pd.DataFrame = None
	model: nn.Model = None
	isDataset: bool = False
	isModel: bool = False

	print("	- Multilayer Perceptron - \n")
	print("you can exit by typing : 'exit' or 'quit' at anytime\n")
	while True:
		try:
			print("--- choose phase : --")
			print("1) preprocess dataset")
			if dataset is not None:
				print("2) train neural network")
				if (model is not None):
					print("3) predict")
			print("		---------")
			#try:
			display_str = "(1, 2, 3): " if isDataset and isModel else "(1, 2): " if isDataset else "(1): " 
			user_input = input(f"Choose an option {display_str }")
			if (user_input == "exit" or user_input == "quit"):
				break
			option_choosed = int(user_input)
			display_str = "Enter dataset path: " if option_choosed == 1 else "Enter the json model path: "
			path = ""
			if (option_choosed == 1 or (option_choosed == 2 and isDataset)):
				path = str(input(display_str))

			if (path == "exit" or user_input == "quit"):
				break

			if (option_choosed == 1):
				#dataset = (train_X, train_Y), (test_X, test_Y) = create_data(1000, 2)
				dataset = (train_X, train_Y), (test_X, test_Y) = preprocess_binary_output_data(extract_csv(path))
				isDataset = True
				print(f"train shape : {train_X.shape}, {train_Y.shape}, test shape : {test_X.shape}, {test_Y.shape}")
				#if (test_Y.shape[1] == 1)
			elif option_choosed == 2 and isDataset:
				model_data = nn.parse_model_json(path)
				model = nn.compile_and_fit_parsed_model(model_data, preprocess_func=None, data=dataset)
				isModel = True
			elif (option_choosed == 3 and isDataset and isModel):
				model.evaluate(test_X, test_Y)
			else:
				print("Invalid input.")
		except Exception as e:
			print("Error : ", e)



	#model = nn.Model()
	
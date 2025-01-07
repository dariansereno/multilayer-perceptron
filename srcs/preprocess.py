import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tools import extract_csv

tolerance = 0.5

class PreprocessingError(Exception):
	pass

def standardize_data(data):
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	standardized_data = (data - mean) / std
	return standardized_data

def convert_target_column(df: pd.DataFrame):
	target_columns = [col for col, val in df.nunique().items() if val ==  2]

	if len(target_columns) ==  1:
		target_column = target_columns[0]
	else:
		raise PreprocessingError("Too much target columns")

	df[0], df[target_column] = df[target_column].copy(), df[0].copy()
	le = LabelEncoder()
	df[0] = le.fit_transform(df[0])
	df.rename(columns={0: "predict"}, inplace=True)
	return df

def compute_correlation(x, y):
	x_mean = np.mean(x)
	y_mean = np.mean(y)

	x_diff = x - x_mean
	y_diff = y - y_mean

	numerator = np.sum(x_diff * y_diff)

	denominator = np.sqrt(np.sum(x_diff**2) * np.sum(y_diff**2))
	return (numerator / denominator)

#Ne sert a rien car supprimes les valeurs qui n'ont pas de relations linéaire alors que le modele
#peut trouver d'autre type de relation
def drop_unrelated_columns(df: pd.DataFrame):
	for i in range(len(df.columns)):
		if (i != 0):
			correlation = compute_correlation(df[0], df[i])
			if (correlation < tolerance):
				df.drop(i, axis=1, inplace=True)
	exit()
	label_map = {old: new for new, old in enumerate(df.columns)}
	label_map[0] = "predict"
	df = df.rename(columns=label_map)
	return df

def split_dataset(df: pd.DataFrame):
	df = df.sample(frac=1).reset_index(drop=True)
	
	split_percentage = 0.8
	index = int(len(df) * split_percentage)
	Y = df["predict"]
	df = df.drop("predict", axis=1)

	y = np.array(Y)
	x = np.array(df)
	indices = np.random.permutation(len(x))
	y = y[indices]
	x = x[indices]

	return (x[:index],x[index:]),(y[:index], y[index:])


def preprocess_binary_output_data(data: str | pd.DataFrame):
	if (isinstance(data, str)):
		df = extract_csv(data)
	elif (isinstance(data, pd.DataFrame)):
		df = data
	else:
		raise ValueError("preprocess_binary_output_data(): Invalid type")

	df = convert_target_column(df)
	#df = drop_unrelated_columns(df)
	without_pred_df = df.drop("predict", axis=1)
	df[df.columns[1:]] = standardize_data(without_pred_df)

	(train_X, test_X), (train_Y, test_Y) = split_dataset(df)
	if (len(test_Y) == test_X.shape[0]):
		train_Y = train_Y.reshape(-1, 1)
		test_Y = test_Y.reshape(-1, 1)
	return (train_X, train_Y), (test_X, test_Y)

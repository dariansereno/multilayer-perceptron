import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tools import extract_csv


def show_df(df: pd.DataFrame):
  print(df)

def scatter_diagram(df: pd.DataFrame):
  numeric_columns = df.drop(1, axis=1)
  corr_matrix = numeric_columns.corr()
  sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
  plt.show()

def box_plot(df: pd.DataFrame, feature: int):
	sns.boxplot(x=1, y=feature, data=df)
	plt.show()

def cat_plot(df: pd.DataFrame, feature: int):
	sns.catplot(x=1, y=feature, data=df)
	plt.show()

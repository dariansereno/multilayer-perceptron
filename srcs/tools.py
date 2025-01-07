import pandas as pd
import numpy as np

def extract_csv(filepath: str):
  return pd.read_csv(filepath, header=None)

def to_one_hot(y, num_classes=2):
  if (y.ndim == 2):
    y = y.flatten()
  return np.eye(num_classes)[y]

__all__ = ["to_one_hot", "extract_csv"]
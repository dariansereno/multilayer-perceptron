import pandas as pd

def extract_csv(filepath: str):
  try:
    return pd.read_csv(filepath, header=None)
  except:
    raise FileNotFoundError
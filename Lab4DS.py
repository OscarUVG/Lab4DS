import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")

print("--- Vista inicial de los datos ---")
print(train.head())

print("--- Conteo de valores NA ---")
print(train.isnull().sum())



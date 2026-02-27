import numpy as np
import pandas as pd
import sklearn.model_selection as ms

dataset = pd.read_csv("../datasets/Healthcare_Data_Analysis_for_readmission.csv")
print(dataset.head())
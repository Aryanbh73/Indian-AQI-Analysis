# Step 1: Dataset creation and overview
import pandas as pd
import numpy as np

df = pd.read_csv('data/city_day.csv')

print(df.shape)
print(df.head())
print(df.info())
print(df.dtypes)
print(df.describe())

# Step 2: Handling missing values, redundancy -> data cleaning]

# Check missing values
print(df.isnull().sum())
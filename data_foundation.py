# Step 1: Dataset creation and overview
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

df = pd.read_csv('data/city_day.csv')

print(df.shape)
print(df.head())
print(df.info())
print(df.dtypes)
print(df.describe())

# Step 2: Handling missing values, redundancy -> data cleaning]

# Check missing values
print(df.isnull().sum())

num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())  

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0])

print("Duplicated:", df.duplicated().sum())
df = df.drop_duplicates()

df.to_csv('data/aqi_cleaned.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Extract useful date features
df['Year']  = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day']   = df['Date'].dt.day

# Drop original Date column
df = df.drop(columns=['Date'])

print(df.head())

# Label Encoding for 'City'
le = LabelEncoder()
df['City_encoded'] = le.fit_transform(df['City'])

# One-Hot Encoding for 'AQI_Bucket'
df = pd.get_dummies(df, columns=['AQI_Bucket'], drop_first=True)

print(df.head())
print(df.shape)

# Range, Variance, Std Dev, IQR
for col in ['PM2.5', 'PM10', 'AQI']:
    print(f"\n--- {col} ---")
    print(f"Range    : {df[col].max() - df[col].min():.2f}")
    print(f"Variance : {df[col].var():.2f}")
    print(f"Std Dev  : {df[col].std():.2f}")
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    print(f"IQR      : {Q3 - Q1:.2f}")

# Bin AQI into categories
df['AQI_Category'] = pd.cut(
    df['AQI'],
    bins=[0, 50, 100, 200, 300, 400, 500],
    labels=['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
)

print(df['AQI_Category'].value_counts())


scale_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'AQI']

df_standardized = df.copy()
df_standardized[scale_cols] = StandardScaler().fit_transform(df[scale_cols])

df_normalized = df.copy()
df_normalized[scale_cols] = MinMaxScaler().fit_transform(df[scale_cols])

print("Standardized:\n", df_standardized[scale_cols].describe())
print("\nNormalized:\n",  df_normalized[scale_cols].describe())

# Section 8: Outlier Detection and Removal — Z-Score Method

z_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
          'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']

# Keep only columns that actually exist in the dataframe
z_cols = [c for c in z_cols if c in df.columns]

# Compute absolute Z-scores for each selected column
z_scores = pd.DataFrame(
    np.abs(stats.zscore(df[z_cols].fillna(df[z_cols].median()))),
    columns=z_cols,
    index=df.index
)

# Threshold
Z_THRESHOLD = 3

# Per-column outlier counts
print("=== Z-Score Outlier Counts (|Z| > 3) per Column ===")
col_outlier_counts = (z_scores > Z_THRESHOLD).sum()
print(col_outlier_counts.to_string())

# Mask: rows where ANY column exceeds the threshold
outlier_mask = (z_scores > Z_THRESHOLD).any(axis=1)
print(f"\nTotal rows flagged as outliers : {outlier_mask.sum()}")
print(f"Total rows before removal      : {len(df)}")

# Remove outlier rows
df = df[~outlier_mask].reset_index(drop=True)
print(f"Total rows after removal       : {len(df)}")

# Statistical Analysis: T-test for AQI between Delhi and Mumbai
delhi = df[df['City'] == 'Delhi']['AQI']
mumbai = df[df['City'] == 'Mumbai']['AQI']
t_stat, p_val = stats.ttest_ind(delhi.dropna(), mumbai.dropna())
print(f"T-stat: {t_stat:.4f}, P-value: {p_val:.4f}")
print("Significant difference!" if p_val < 0.05 else "No significant difference")

# Skewness and Kurtosis
for col in ['PM2.5', 'PM10', 'AQI', 'NO2']:
    skew = df[col].skew()
    kurt = df[col].kurtosis()
    print(f"{col:10s} | Skewness: {skew:7.4f} | Kurtosis: {kurt:7.4f}")

if skew > 0:
    print("Data is right skewed.")
elif skew < 0:
    print("Data is left skewed.")
if kurt > 3:
    print("Data is leptokurtic (heavy tails).")
elif kurt < 3:
    print("Data is platykurtic (light tails).")

for col in ['PM2.5', 'PM10', 'AQI']:
    print(f"\n--- {col} ---")
    print(f"Mean   : {df[col].mean():.2f}")
    print(f"Median : {df[col].median():.2f}")
    print(f"Mode   : {df[col].mode()[0]:.2f}")

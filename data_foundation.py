# Step 1: Dataset creation and overview
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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

# Section 5 Visualization: AQI Category Bar Chart
cat_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
cat_counts = df['AQI_Category'].value_counts().reindex(cat_order)

plt.figure(figsize=(9, 5))
sns.barplot(x=cat_counts.index, y=cat_counts.values, color='skyblue')
plt.title('AQI Category Distribution')
plt.xlabel('AQI Category')
plt.ylabel('Number of Records')
plt.xticks(rotation=20)
plt.show()

scale_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'AQI']

df_standardized = df.copy()
df_standardized[scale_cols] = StandardScaler().fit_transform(df[scale_cols])

df_normalized = df.copy()
df_normalized[scale_cols] = MinMaxScaler().fit_transform(df[scale_cols])

print("Standardized:\n", df_standardized[scale_cols].describe())
print("\nNormalized:\n",  df_normalized[scale_cols].describe())

# Section 6 Visualization: Original vs Standardized vs Normalized
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.histplot(df['PM2.5'], bins=30, ax=axes[0])
axes[0].set_title('Original PM2.5')

sns.histplot(df_standardized['PM2.5'], bins=30, ax=axes[1])
axes[1].set_title('Standardized PM2.5')

sns.histplot(df_normalized['PM2.5'], bins=30, ax=axes[2])
axes[2].set_title('Normalized PM2.5')

plt.show()

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

# Section 8 Visualization: Box Plot and Outlier Count Bar Chart
plot_cols_outlier = [c for c in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'AQI'] if c in df.columns]

# Box plots
df_melt = df[plot_cols_outlier].melt(var_name='Pollutant', value_name='Value')
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_melt, x='Pollutant', y='Value')
plt.title('Outliers in Pollutant Columns')
plt.xlabel('Pollutant')
plt.ylabel('Value')
plt.show()

# Outlier count bar chart
sorted_counts = col_outlier_counts.sort_values(ascending=False).reset_index()
sorted_counts.columns = ['Column', 'Outliers']
plt.figure(figsize=(9, 5))
sns.barplot(data=sorted_counts, x='Outliers', y='Column', color='salmon')
plt.title('Z-Score Outlier Count')
plt.xlabel('Number of Outliers')
plt.ylabel('Column')
plt.show()

# Remove outlier rows
df = df[~outlier_mask].reset_index(drop=True)
print(f"Total rows after removal       : {len(df)}")

# Statistical Analysis: T-test for AQI between Delhi and Mumbai
delhi = df[df['City'] == 'Delhi']['AQI']
mumbai = df[df['City'] == 'Mumbai']['AQI']
t_stat, p_val = stats.ttest_ind(delhi.dropna(), mumbai.dropna())
print(f"T-stat: {t_stat:.4f}, P-value: {p_val:.4f}")
print("Significant difference!" if p_val < 0.05 else "No significant difference")

# Section 9 Visualization: Delhi vs Mumbai
aqi_city = df[df['City'].isin(['Delhi', 'Mumbai'])][['City', 'AQI']].dropna()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(data=aqi_city, x='AQI', hue='City', bins=30, ax=axes[0])
axes[0].set_title('AQI Distribution: Delhi vs Mumbai')
axes[0].set_xlabel('AQI')
axes[0].set_ylabel('Count')

sns.boxplot(data=aqi_city, x='City', y='AQI', ax=axes[1])
axes[1].set_title('AQI Box Plot: Delhi vs Mumbai')
axes[1].set_xlabel('City')
axes[1].set_ylabel('AQI')
plt.show()

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

# Section 10 Visualization: Simple distributions
dist_cols = [c for c in ['PM2.5', 'PM10', 'AQI', 'NO2'] if c in df.columns]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, col in zip(axes.flatten(), dist_cols):
    sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title(f'{col} Distribution')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree

df = pd.read_csv('data/aqi_cleaned.csv')

city_stats = df.groupby('City')['AQI'].agg(['mean', 'median', 'std', 'min', 'max'])
print(city_stats.sort_values('mean').head(10))

yearly = df.groupby('Year')[['AQI','PM2.5', 'PM10']].mean()
print(yearly)

city_stats['mean'].sort_values().plot(kind='bar', figsize=(12,6), title='Average AQI by City')
plt.show()

num = df.select_dtypes(include=[np.number])
corr_matrix = num.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix - Indian AQI Dataset')
plt.show()
print(corr_matrix['AQI'].sort_values())

features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
target   = 'AQI'

model_df = df[features + [target]].dropna()
X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = model.score(X_test, y_test)
print("R2 score:",r2)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

for feat, coef in zip(features, model.coef_):
    print(f"  {feat:10s}: {coef:.4f}")

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Linear Regression: Actual vs Predicted')
plt.show()

df['Polluted'] = (df['AQI'] > 100).astype(int)
X_clf = df[features].dropna()
y_clf = df.loc[X_clf.index, 'Polluted']
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
print("Logistic Regression:")
print(f"Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression — Confusion Matrix')
plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("K-Nearest Neighbour:")
print(f"Accuracy : {accuracy_score(y_test, y_pred_knn):.4f}")
print(classification_report(y_test, y_pred_knn))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Greens')
plt.title('KNN — Confusion Matrix')
plt.show()

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree:")
print(f"Accuracy : {accuracy_score(y_test, y_pred_dt):.4f}")
print(classification_report(y_test, y_pred_dt))

plt.figure(figsize=(15, 10))
plot_tree(dt, feature_names=features, class_names=['Clean','Polluted'],filled=True, rounded=True, fontsize=9)
plt.title('Decision Tree Visualization')
plt.show()

results = {
    'Logistic Regression': accuracy_score(y_test, y_pred_lr),
    'KNN'                : accuracy_score(y_test, y_pred_knn),
    'Decision Tree'      : accuracy_score(y_test, y_pred_dt),
}
plt.figure(figsize=(7,4))
plt.bar(results.keys(), results.values(), color=['steelblue','seagreen','tomato'])
plt.ylim(0.8, 1.0)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

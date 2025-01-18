from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load California Housing dataset
california = fetch_california_housing(as_frame=True)
df = california.frame

# Rename target column for consistency
df.rename(columns={'MedHouseVal': 'PRICE'}, inplace=True)

# Display first few rows
print("Dataset Head:\n", df.head())

# Step 2: Data Preprocessing
# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(df.drop(columns='PRICE'))
y = df['PRICE']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Exploratory Data Analysis (EDA)
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Pairplot for selected features
sns.pairplot(df[['PRICE', 'MedInc', 'AveRooms', 'AveOccup']])
plt.title("pair plot")
plt.show()

# Step 4: Model Building
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Ridge Regression
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)

# Predictions
y_pred_lin = lin_reg.predict(X_test)
y_pred_ridge = ridge_reg.predict(X_test)

# Step 5: Model Evaluation
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.2f}, R2: {r2:.2f}")

evaluate_model(y_test, y_pred_lin, "Linear Regression")
evaluate_model(y_test, y_pred_ridge, "Ridge Regression")

# Visualize Predictions vs Actual Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lin, label='Linear Regression', alpha=0.6)
plt.scatter(y_test, y_pred_ridge, label='Ridge Regression', alpha=0.6)
plt.plot([0, 5], [0, 5], '--r', label='Perfect Fit')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Prediction vs Actual Prices")
plt.legend()
plt.show()

# Step 6: Conclusion
print("Linear Regression and Ridge Regression models evaluated. Choose based on performance metrics.")

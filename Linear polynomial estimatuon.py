import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic dataset
np.random.seed(42)
data_size = 100
X = np.random.rand(data_size, 1) * 10  # Feature: random values between 0 and 10
y = 2 * (X ** 2) + 3 * X + 5 + np.random.randn(data_size, 1) * 5  # Target: quadratic relationship with some noise

# Create DataFrame
df = pd.DataFrame(np.hstack((X, y)), columns=['Feature', 'Target'])
print(df.head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# Polynomial Regression (degree=2)
poly_features = PolynomialFeatures(degree=2)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)
y_pred_poly = poly_reg.predict(X_poly_test)

# Evaluate the models
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("\nLinear Regression:")
print("Mean Squared Error:", mse_lin)
print("R-squared:", r2_lin)

print("\nPolynomial Regression (degree=2):")
print("Mean Squared Error:", mse_poly)
print("R-squared:", r2_poly)

# Visualization
plt.figure(figsize=(14, 7))

# Scatter plot of the original data
plt.scatter(X, y, color='gray', label='Data')

# Plot Linear Regression results
plt.plot(X_test, y_pred_lin, color='blue', label='Linear Regression', linewidth=2)

# Plot Polynomial Regression results
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
X_poly_plot = poly_features.transform(X_plot)
y_poly_plot = poly_reg.predict(X_poly_plot)
plt.plot(X_plot, y_poly_plot, color='red', label='Polynomial Regression (degree=2)', linewidth=2)

plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear vs Polynomial Regression')
plt.legend()
plt.show()

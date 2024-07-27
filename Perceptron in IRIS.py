import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report
0
# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Display the first few rows of the dataset
print("Features:\n", X[:5])
print("Labels:\n", y[:5])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Perceptron
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=42)

# Train the Perceptron
ppn.fit(X_train, y_train)

# Predict the labels on the test set
y_pred = ppn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

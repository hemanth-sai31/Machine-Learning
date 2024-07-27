import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
def load_data(file_path):
    return pd.read_csv(file_path)
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
def knn_classify(training_data, training_labels, test_instance, k):
    distances = []
    for index, instance in training_data.iterrows():
        distance = euclidean_distance(instance.values, test_instance.values)
        distances.append((distance, training_labels[index]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    neighbor_labels = [label for _, label in neighbors]
    most_common = Counter(neighbor_labels).most_common(1)
    return most_common[0][0]
def train_test_knn(data, target_column, k, test_size=0.2):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    predictions = []
    for _, test_instance in X_test.iterrows():
        prediction = knn_classify(X_train, y_train, test_instance, k)
        predictions.append(prediction)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy
def classify_new_sample(training_data, training_labels, new_sample, k):
    return knn_classify(training_data, training_labels, new_sample, k)
file_path = 'iris.csv'
data = load_data(file_path)
target_column = 'species'  k = 3
accuracy = train_test_knn(data, target_column, k)
print(f"Accuracy: {accuracy:.2f}")
new_sample = pd.Series({'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2})  # Replace with actual values
classification = classify_new_sample(data.drop(columns=[target_column]), data[target_column], new_sample, k)
print(f"Classification: {classification}")

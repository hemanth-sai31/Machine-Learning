import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Generate synthetic dataset
np.random.seed(42)
data_size = 1000

sizes = np.random.randint(500, 3500, data_size)  # in square feet
locations = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], data_size)
bedrooms = np.random.randint(1, 6, data_size)
bathrooms = np.random.randint(1, 4, data_size)
ages = np.random.randint(0, 50, data_size)  # age of the house
prices = (50000 + sizes * 150 + bedrooms * 10000 + bathrooms * 5000 - ages * 200 + np.random.normal(0, 10000, data_size)).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'size': sizes,
    'location': locations,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': ages,
    'price': prices
})

# Display the first few rows of the dataset
print(df.head())

# Split data into features (X) and target (y)
X = df.drop('price', axis=1)
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
# Define the preprocessing steps for numerical and categorical features
numerical_features = ['size', 'bedrooms', 'bathrooms', 'age']
categorical_features = ['location']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R-squared:", r2_score(y_test, y_pred))

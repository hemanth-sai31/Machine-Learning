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

years = np.random.randint(2000, 2022, data_size)
mileages = np.random.randint(5000, 200000, data_size)
brands = np.random.choice(['Toyota', 'Ford', 'BMW', 'Audi', 'Mercedes'], data_size)
engine_sizes = np.random.uniform(1.0, 5.0, data_size)
prices = (30000 - (2022 - years) * 1000 + engine_sizes * 3000 - mileages * 0.1 + np.random.normal(0, 2000, data_size)).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'year': years,
    'mileage': mileages,
    'brand': brands,
    'engine_size': engine_sizes,
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
numerical_features = ['year', 'mileage', 'engine_size']
categorical_features = ['brand']

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

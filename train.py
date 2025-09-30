from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load data
df = pd.read_csv('insurance.csv')

# We'll predict 'charges' directly, but could also predict log_charges and convert back
X = df.drop('charges', axis=1)
y = df['charges']

# Identify categorical and numerical features
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# In train.py, continued...

# Define the model
# XGBoost is a great choice for this kind of tabular data
model = xgb.XGBRegressor(objective='reg:squarederror',
                         n_estimators=100,
                         learning_rate=0.1,
                         max_depth=4,
                         random_state=42)

# Create the full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance on Test Set:")
# On average, our prediction is off by this amount.
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
# The model explains this % of variance in the data.
print(f"R-squared (R2): {r2:.2f}")

# Save the final trained pipeline
joblib.dump(pipeline, 'hsa_predictor_pipeline.pkl')
print("\nModel pipeline saved to hsa_predictor_pipeline.pkl")

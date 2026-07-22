from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "insurance.csv"
MODEL_PATH = BASE_DIR / "hsa_predictor_pipeline.pkl"

# Load data
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["charges"])
y = df["charges"]

categorical_features = ["sex", "smoker", "region"]
numerical_features = ["age", "bmi", "children"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance on Test Set:")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"R-squared (R2): {r2:.2f}")

joblib.dump(pipeline, MODEL_PATH)
print(f"\nModel pipeline saved to {MODEL_PATH}")

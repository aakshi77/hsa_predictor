from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hsa_predictor_pipeline.pkl"
DATA_PATH = BASE_DIR / "insurance.csv"

app = Flask(__name__)


def build_fallback_pipeline():
    """Train a simple scikit-learn pipeline when the saved model is unavailable."""
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
    regressor = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_depth=8,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def load_pipeline():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as exc:
            app.logger.warning("Unable to load saved pipeline, training a fallback model: %s", exc)
    return build_fallback_pipeline()


pipeline = load_pipeline()


def build_input_frame(data):
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object with prediction inputs.")

    age_map = {"18-24": 21, "25-34": 30, "35-44": 40, "45-54": 50, "55-64": 60}
    enrollment_map = {"Individual": 0, "Individual + Spouse": 1, "Family": 2}
    chronic_map = {"No": "no", "Yes": "yes"}

    age_bracket = data.get("ageBracket") or data.get("age_bracket")
    if age_bracket not in age_map:
        raise ValueError("ageBracket must be one of: 18-24, 25-34, 35-44, 45-54, 55-64")

    children = data.get("children")
    if children is None:
        children = enrollment_map.get(data.get("enrollingAs"), 0)

    smoker = data.get("smoker")
    if smoker is None:
        smoker = chronic_map.get(data.get("chronicCondition"), "no")
    smoker = str(smoker).lower()

    input_data = {
        "age": age_map[age_bracket],
        "bmi": float(data.get("bmi", 25.0)),
        "children": int(children),
        "sex": str(data.get("sex", "female")).lower(),
        "smoker": smoker,
        "region": str(data.get("region", "northwest")).lower(),
    }
    return pd.DataFrame([input_data])


@app.route("/predict", methods=["POST"])
def predict():
    """Receives user input, predicts annual healthcare expenses, and returns an HSA recommendation."""
    try:
        data = request.get_json(silent=True) or {}
        input_df = build_input_frame(data)

        prediction = float(pipeline.predict(input_df)[0])

        mae = 2500.0
        lower_bound = max(0.0, prediction - mae / 2)
        upper_bound = prediction + mae / 2

        enrollment = data.get("enrollingAs", "Individual")
        enrollment_map = {"Individual": 0, "Individual + Spouse": 1, "Family": 2}
        hsa_max_individual = 4300
        hsa_max_family = 8550
        tax_rate = 0.22

        hsa_max = hsa_max_family if enrollment_map.get(enrollment, 0) > 0 else hsa_max_individual
        tax_savings = hsa_max * tax_rate
        recommended_contribution = round(prediction / 100) * 100

        response = {
            "predictionRange": f"${int(lower_bound):,} - ${int(upper_bound):,}",
            "recommendation": f"We recommend an annual HSA contribution of at least ${int(recommended_contribution):,}.",
            "taxSavings": f"A maximum contribution of ${hsa_max:,} could save you an estimated ${int(tax_savings):,} in taxes.",
        }
        return jsonify(response)

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"An error occurred: {exc}"}), 500


@app.route("/")
def home():
    """Serves the main HTML page when it exists, otherwise returns a simple fallback page."""
    index_path = BASE_DIR / "static" / "index.html"
    if index_path.exists():
        return send_from_directory(str(index_path.parent), index_path.name)

    return """<html><body><h1>HSA Predictor</h1><p>The API is running. Use POST /predict.</p></body></html>"""


if __name__ == "__main__":
    app.run(debug=True)

# In app.py
from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd

# 1. Initialize the Flask app and load the model
app = Flask(__name__)
pipeline = joblib.load('hsa_predictor_pipeline.pkl')


# 2. Define the API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    """Receives user input, predicts expenses, and returns a recommendation."""
    data = request.get_json()

    try:
        # Map user-friendly inputs to the model's expected feature format
        age_map = {'18-24': 21, '25-34': 30,
                   '35-44': 40, '45-54': 50, '55-64': 60}
        enrollment_map = {'Individual': 0,
                          'Individual + Spouse': 1, 'Family': 3}
        chronic_map = {'No': 'no', 'Yes': 'yes'}

        input_data = {
            'age': age_map[data['ageBracket']],
            'bmi': 25,  # Using a default BMI for simplicity
            'children': enrollment_map[data['enrollingAs']],
            'sex': 'female',  # Using a default sex for simplicity
            'smoker': chronic_map[data['chronicCondition']],
            'region': data['region'].lower()
        }
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = pipeline.predict(input_df)[0]

        # Calculate a simple confidence range using the Mean Absolute Error (MAE)
        mae = 4181  # NOTE: Use the actual MAE from your model training
        lower_bound = max(0, prediction - mae / 2)
        upper_bound = prediction + mae / 2

        # Generate financial recommendations
        hsa_max_individual = 4300  # For 2025
        hsa_max_family = 8550  # For 2025
        tax_rate = 0.22  # Assumed tax bracket

        hsa_max = hsa_max_family if enrollment_map[data['enrollingAs']
                                                   ] > 0 else hsa_max_individual
        tax_savings = hsa_max * tax_rate
        recommended_contribution = round(prediction / 100) * 100

        # Format the JSON response
        response = {
            'predictionRange': f"${int(lower_bound):,} - ${int(upper_bound):,}",
            'recommendation': f"We recommend an annual HSA contribution of at least ${int(recommended_contribution):,}.",
            'taxSavings': f"A maximum contribution of ${hsa_max:,} could save you an estimated ${int(tax_savings):,} in taxes."
        }
        return jsonify(response)

    except KeyError as e:
        return jsonify({'error': f"Missing input data: {str(e)}"}), 400
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500


# 3. Define the route for the homepage to serve the HTML file
@app.route('/')
def home():
    """Serves the main HTML page."""
    return send_from_directory('static', 'index.html')


# 4. Run the app
# This block should always be at the very end of the file
if __name__ == '__main__':
    app.run(debug=True)

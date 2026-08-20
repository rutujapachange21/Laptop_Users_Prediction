import os
import pickle
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained SVC model
MODEL_PATH = "model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Embedded HTML & CSS Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Prediction Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }
        .card {
            background: #ffffff;
            width: 100%;
            max-width: 520px;
            border-radius: 16px;
            padding: 36px 32px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.25), 0 8px 10px -6px rgba(0, 0, 0, 0.25);
        }
        .header {
            text-align: center;
            margin-bottom: 28px;
        }
        .header h1 {
            font-size: 24px;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 6px;
        }
        .header p {
            font-size: 14px;
            color: #64748b;
        }
        .form-group {
            margin-bottom: 18px;
        }
        .form-row {
            display: flex;
            gap: 16px;
        }
        .form-row .form-group {
            flex: 1;
        }
        label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            color: #334155;
            margin-bottom: 6px;
        }
        input, select {
            width: 100%;
            padding: 10px 14px;
            font-size: 14px;
            color: #0f172a;
            border: 1.5px solid #cbd5e1;
            border-radius: 8px;
            background-color: #f8fafc;
            transition: all 0.2s ease;
            outline: none;
        }
        input:focus, select:focus {
            border-color: #6366f1;
            background-color: #ffffff;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
        }
        button {
            width: 100%;
            padding: 12px;
            font-size: 15px;
            font-weight: 600;
            color: #ffffff;
            background-color: #4f46e5;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s ease;
            margin-top: 10px;
        }
        button:hover {
            background-color: #4338ca;
        }
        .result-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
            animation: fadeIn 0.3s ease-in-out;
        }
        .result-yes {
            background-color: #ecfdf5;
            border: 1.5px solid #a7f3d0;
            color: #065f46;
        }
        .result-no {
            background-color: #fef2f2;
            border: 1.5px solid #fecaca;
            color: #991b1b;
        }
        .result-box h3 {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .result-box .value {
            font-size: 22px;
            font-weight: 700;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <h1>Classification Portal</h1>
            <p>Enter the demographic details below to generate a prediction.</p>
        </div>

        <form method="POST" action="/">
            <div class="form-row">
                <div class="form-group">
                    <label for="Age">Age</label>
                    <input type="number" id="Age" name="Age" step="any" placeholder="e.g. 35" required>
                </div>
                <div class="form-group">
                    <label for="Gender">Gender</label>
                    <input type="number" id="Gender" name="Gender" step="any" placeholder="Encoded value (e.g. 0/1)" required>
                </div>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label for="Region">Region</label>
                    <input type="number" id="Region" name="Region" step="any" placeholder="Encoded value" required>
                </div>
                <div class="form-group">
                    <label for="Occupation">Occupation</label>
                    <input type="number" id="Occupation" name="Occupation" step="any" placeholder="Encoded value" required>
                </div>
            </div>

            <div class="form-group">
                <label for="Income">Annual Income</label>
                <input type="number" id="Income" name="Income" step="any" placeholder="e.g. 50000" required>
            </div>

            <button type="submit">Predict Outcome</button>
        </form>

        {% if prediction is not none %}
        <div class="result-box result-{{ prediction|lower }}">
            <h3>Prediction Result</h3>
            <div class="value">{{ prediction|upper }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        features = {
            "Age": float(request.form.get("Age", 0)),
            "Gender": float(request.form.get("Gender", 0)),
            "Region": float(request.form.get("Region", 0)),
            "Occupation": float(request.form.get("Occupation", 0)),
            "Income": float(request.form.get("Income", 0)),
        }
        input_df = pd.DataFrame([features])
        raw_pred = model.predict(input_df)[0]
        prediction = str(raw_pred)

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

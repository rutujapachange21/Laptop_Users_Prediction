import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load SVM Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelsvm.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Inline HTML/CSS Template for single-file deployment
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVM Prediction Portal</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --success-bg: #ecfdf5;
            --success-text: #065f46;
            --danger-bg: #fef2f2;
            --danger-text: #991b1b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        }

        body {
            background-color: var(--bg);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }

        .container {
            width: 100%;
            max-width: 520px;
            background: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.04);
            padding: 36px;
            border: 1px solid var(--border);
        }

        .header {
            text-align: center;
            margin-bottom: 28px;
        }

        .header h1 {
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 6px;
        }

        .header p {
            font-size: 0.9rem;
            color: var(--text-muted);
        }

        .form-group {
            margin-bottom: 18px;
        }

        label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-main);
        }

        input, select {
            width: 100%;
            padding: 10px 14px;
            border-radius: 8px;
            border: 1.5px solid var(--border);
            background: #fff;
            font-size: 0.95rem;
            color: var(--text-main);
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        button {
            width: 100%;
            background: var(--primary);
            color: #fff;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s ease, transform 0.05s ease;
            margin-top: 10px;
        }

        button:hover {
            background: var(--primary-hover);
        }

        button:active {
            transform: scale(0.99);
        }

        .result-box {
            margin-top: 24px;
            padding: 14px;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            font-size: 1.05rem;
        }

        .result-yes {
            background-color: var(--success-bg);
            color: var(--success-text);
            border: 1px solid #a7f3d0;
        }

        .result-no {
            background-color: var(--danger-bg);
            color: var(--danger-text);
            border: 1px solid #fecaca;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SVM Model Predictor</h1>
            <p>Enter individual attributes below to generate a prediction</p>
        </div>

        <form method="POST" action="/predict">
            <div class="grid-2">
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" required min="1" max="120" placeholder="e.g. 32" value="{{ request.form.get('age', '') }}">
                </div>
                <div class="form-group">
                    <label for="gender">Gender</label>
                    <select id="gender" name="gender" required>
                        <option value="1" {% if request.form.get('gender') == '1' %}selected{% endif %}>Male</option>
                        <option value="0" {% if request.form.get('gender') == '0' %}selected{% endif %}>Female</option>
                    </select>
                </div>
            </div>

            <div class="grid-2">
                <div class="form-group">
                    <label for="region">Region Code</label>
                    <input type="number" id="region" name="region" required placeholder="e.g. 0 or 1" value="{{ request.form.get('region', '') }}">
                </div>
                <div class="form-group">
                    <label for="occupation">Occupation Code</label>
                    <input type="number" id="occupation" name="occupation" required placeholder="e.g. 1" value="{{ request.form.get('occupation', '') }}">
                </div>
            </div>

            <div class="form-group">
                <label for="income">Income ($)</label>
                <input type="number" step="any" id="income" name="income" required placeholder="e.g. 45000" value="{{ request.form.get('income', '') }}">
            </div>

            <button type="submit">Run Prediction</button>
        </form>

        {% if prediction is not none %}
        <div class="result-box {% if prediction|lower == 'yes' %}result-yes{% else %}result-no{% endif %}">
            Result: <strong>{{ prediction|upper }}</strong>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        gender = float(request.form['gender'])
        region = float(request.form['region'])
        occupation = float(request.form['occupation'])
        income = float(request.form['income'])

        features = np.array([[age, gender, region, occupation, income]])
        prediction = model.predict(features)[0]

        return render_template_string(HTML_TEMPLATE, prediction=str(prediction))
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

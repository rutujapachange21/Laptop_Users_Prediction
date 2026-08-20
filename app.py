import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load SVM Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelsvm.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Modern, Glassmorphic UI with Clear Labels and Tooltips
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Prediction Intelligence Portal</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            --accent: #ec4899;
            --bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --input-bg: rgba(15, 23, 42, 0.6);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-hover: rgba(99, 102, 241, 0.4);
            --success-glow: rgba(16, 185, 129, 0.15);
            --success-border: #10b981;
            --danger-glow: rgba(239, 68, 68, 0.15);
            --danger-border: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.12) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 32px 16px;
        }

        .portal-wrapper {
            width: 100%;
            max-width: 580px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 24px;
            border: 1px solid var(--card-border);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            padding: 40px;
            position: relative;
            overflow: hidden;
        }

        .portal-wrapper::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #6366f1, #ec4899, #6366f1);
            background-size: 200% 100%;
            animation: gradientMove 6s ease infinite;
        }

        @keyframes gradientMove {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .header {
            text-align: center;
            margin-bottom: 32px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            background: rgba(99, 102, 241, 0.12);
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #818cf8;
            margin-bottom: 12px;
        }

        .badge-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #818cf8;
            box-shadow: 0 0 8px #818cf8;
        }

        .header h1 {
            font-size: 1.75rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #ffffff;
            margin-bottom: 8px;
        }

        .header p {
            font-size: 0.9rem;
            color: var(--text-muted);
            line-height: 1.5;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #cbd5e1;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .hint-text {
            font-size: 0.7rem;
            font-weight: 400;
            color: var(--text-muted);
        }

        .input-container {
            position: relative;
            display: flex;
            align-items: center;
        }

        input, select {
            width: 100%;
            padding: 12px 14px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: var(--input-bg);
            color: #ffffff;
            font-size: 0.95rem;
            font-weight: 500;
            outline: none;
            transition: all 0.2s ease;
        }

        select option {
            background-color: #1e293b;
            color: #ffffff;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
            background: rgba(15, 23, 42, 0.85);
        }

        input::placeholder {
            color: #475569;
        }

        .submit-btn {
            grid-column: span 2;
            margin-top: 12px;
            padding: 14px;
            background: var(--primary-gradient);
            border: none;
            border-radius: 12px;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.35);
        }

        .submit-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.45);
            filter: brightness(1.05);
        }

        .submit-btn:active {
            transform: translateY(0);
        }

        .result-card {
            margin-top: 28px;
            padding: 18px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            animation: fadeIn 0.3s ease forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-yes {
            background: var(--success-glow);
            border: 1px solid var(--success-border);
        }

        .result-no {
            background: var(--danger-glow);
            border: 1px solid var(--danger-border);
        }

        .result-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 2px;
        }

        .result-value {
            font-size: 1.35rem;
            font-weight: 800;
        }

        .result-yes .result-value {
            color: #34d399;
        }

        .result-no .result-value {
            color: #f87171;
        }

        .status-chip {
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .result-yes .status-chip {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
        }

        .result-no .status-chip {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
        }

        @media (max-width: 540px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            .form-group.full-width, .submit-btn {
                grid-column: span 1;
            }
            .portal-wrapper {
                padding: 24px;
            }
        }
    </style>
</head>
<body>
    <div class="portal-wrapper">
        <div class="header">
            <div class="badge">
                <span class="badge-dot"></span> Support Vector Machine (RBF)
            </div>
            <h1>Inference Engine</h1>
            <p>Select accurate demographic and professional metrics to evaluate the prediction outcome.</p>
        </div>

        <form method="POST" action="/predict" class="form-grid">
            <div class="form-group">
                <label for="age">
                    Age 
                    <span class="hint-text">18 – 100 yrs</span>
                </label>
                <div class="input-container">
                    <input type="number" id="age" name="age" required min="18" max="100" placeholder="e.g. 35" value="{{ request.form.get('age', '') }}">
                </div>
            </div>

            <div class="form-group">
                <label for="gender">
                    Gender
                    <span class="hint-text">Select binary</span>
                </label>
                <div class="input-container">
                    <select id="gender" name="gender" required>
                        <option value="1" {% if request.form.get('gender') == '1' %}selected{% endif %}>Male (1)</option>
                        <option value="0" {% if request.form.get('gender') == '0' %}selected{% endif %}>Female (0)</option>
                    </select>
                </div>
            </div>

            <div class="form-group">
                <label for="region">
                    Region Category
                    <span class="hint-text">Standard code</span>
                </label>
                <div class="input-container">
                    <select id="region" name="region" required>
                        <option value="0" {% if request.form.get('region') == '0' %}selected{% endif %}>Region 0 (Urban / Primary)</option>
                        <option value="1" {% if request.form.get('region') == '1' %}selected{% endif %}>Region 1 (Suburban / Secondary)</option>
                        <option value="2" {% if request.form.get('region') == '2' %}selected{% endif %}>Region 2 (Rural / Tertiary)</option>
                    </select>
                </div>
            </div>

            <div class="form-group">
                <label for="occupation">
                    Occupation Sector
                    <span class="hint-text">Class index</span>
                </label>
                <div class="input-container">
                    <select id="occupation" name="occupation" required>
                        <option value="0" {% if request.form.get('occupation') == '0' %}selected{% endif %}>Sector 0 (Executive / Management)</option>
                        <option value="1" {% if request.form.get('occupation') == '1' %}selected{% endif %}>Sector 1 (Professional / Technical)</option>
                        <option value="2" {% if request.form.get('occupation') == '2' %}selected{% endif %}>Sector 2 (Services / Operations)</option>
                        <option value="3" {% if request.form.get('occupation') == '3' %}selected{% endif %}>Sector 3 (Other / Self-employed)</option>
                    </select>
                </div>
            </div>

            <div class="form-group full-width">
                <label for="income">
                    Annual Income
                    <span class="hint-text">Exact value</span>
                </label>
                <div class="input-container">
                    <input type="number" step="any" id="income" name="income" required placeholder="e.g. 65000" value="{{ request.form.get('income', '') }}">
                </div>
            </div>

            <button type="submit" class="submit-btn">Run Analysis</button>
        </form>

        {% if prediction is not none %}
        <div class="result-card {% if prediction|lower == 'yes' %}result-yes{% else %}result-no{% endif %}">
            <div>
                <div class="result-title">Computed Outcome</div>
                <div class="result-value">{{ prediction|upper }}</div>
            </div>
            <div class="status-chip">
                {% if prediction|lower == 'yes' %}Positive Classification{% else %}Negative Classification{% endif %}
            </div>
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

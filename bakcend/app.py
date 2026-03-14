from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib
import random
import numpy as np
import os

app = Flask(__name__)

# CRITICAL: This allows your React frontend to communicate with this Python backend.
# Without this, the browser will block every request.
CORS(app, resources={r"/*": {"origins": "*"}})

# --- DATABASE INITIALIZATION ---
try:
    # Load your hospital dataset
    hospitals_df = pd.read_csv('hospitals_detailed.csv')
    print(f">> SYSTEM_LOG: Database loaded successfully. {len(hospitals_df)} nodes identified.")
except Exception as e:
    print(f">> CRITICAL_ERROR: Could not load hospitals_detailed.csv. Error: {e}")
    # Fallback to prevent crash if file is missing during testing
    hospitals_df = pd.DataFrame()

# --- ML BRAIN INITIALIZATION ---
# We load the predictor. If it's not found, we use a robust heuristic fallback
# so your demo never crashes even if the model file is moved.
MODEL_PATH = 'blood_predictor.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(">> SYSTEM_LOG: ML Predictor [blood_predictor.pkl] online.")
else:
    model = None
    print(">> SYSTEM_LOG: ML Predictor offline. Using Heuristic Fallback Engine.")

# Central Hub for Distance Calculations (Bengaluru Center)
HUB_COORDS = (12.9767, 77.5713)


# --- CORE LOGIC: PREDICTIVE ANALYSIS ---
def analyze_node(hospital, traffic, rain, blood_group):
    """
    Calculates the stability of a single node based on stock,
    ML predicted drain, and environmental stressors.
    """
    try:
        # 1. Spatial Processing
        lat, lon = map(float, hospital['location'].split(','))
        # Basic Euclidean distance to hub (km approximation)
        dist = np.sqrt((lat - HUB_COORDS[0]) ** 2 + (lon - HUB_COORDS[1]) ** 2) * 111

        # 2. Feature Extraction
        is_neg = 1 if 'neg' in blood_group else 0
        pop_density = int(hospital.get('pop_density', 10000))

        # 3. Prediction (ML vs Heuristic)
        if model:
            # Expected input: [traffic, rain, pop_density, distance, is_negative_type]
            pred_drain = float(model.predict([[traffic, rain, pop_density, dist, is_neg]])[0])
        else:
            # High-fidelity fallback logic
            base_drain = (traffic * 4.2) + (rain * 0.15)
            multiplier = 1.5 if is_neg else 1.0
            pred_drain = float(base_drain * multiplier)

        # 4. Inventory Assessment
        current_stock = int(hospital[blood_group])

        # Threshold logic: Nodes near city center or with rare types need higher buffers
        base_threshold = 15
        safety_buffer = (1.6 if dist < 8 else 1.1) * (1.4 if is_neg else 1.0)
        final_threshold = float(base_threshold * safety_buffer)

        # 5. Stability Determination (JSON SAFE CASTING)
        is_critical = bool((current_stock - pred_drain) < final_threshold)

        return {
            "current": current_stock,
            "threshold": int(final_threshold),
            "predicted_drain": float(round(pred_drain, 1)),
            "is_critical": is_critical
        }
    except Exception as e:
        print(f"Error in analysis: {e}")
        return {"current": 0, "threshold": 15, "predicted_drain": 0, "is_critical": False}


# --- ROUTES ---

@app.route('/hospitals', methods=['GET'])
def get_hospitals():
    """Returns the static hospital data for mapping."""
    if hospitals_df.empty:
        return jsonify([])
    return jsonify(hospitals_df.to_dict(orient='records'))


@app.route('/predict_all_nodes', methods=['GET'])
def predict_all():
    """
    Main endpoint for the Dashboard.
    Synchronizes environmental telemetry with node-by-node analysis.
    """
    # Simulate dynamic city conditions
    traffic_index = float(round(random.uniform(0.5, 4.5), 2))
    rain_intensity = int(random.randint(0, 100))

    blood_groups = ['O_pos', 'O_neg', 'A_pos', 'A_neg', 'B_pos', 'B_neg', 'AB_pos', 'AB_neg']

    results = []
    for _, h in hospitals_df.iterrows():
        # Analyze stability for every blood group at this node
        analysis = {bg: analyze_node(h, traffic_index, rain_intensity, bg) for bg in blood_groups}

        results.append({
            "id": int(h['id']),
            "name": str(h['name']),
            "analysis": analysis
        })

    return jsonify({
        "status": "success",
        "telemetry": {
            "traffic": traffic_index,
            "rain": rain_intensity,
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")
        },
        "nodes": results
    })


@app.route('/simulate_crisis', methods=['POST'])
def simulate_crisis():
    """
    Special endpoint for the Presentation.
    Artificially forces system instability to demonstrate AI responsiveness.
    """
    data = request.json
    scenario = data.get('scenario', 'NORMAL')

    if scenario == 'MONSOON':
        return jsonify({"traffic": 4.8, "rain": 98, "msg": "EMERGENCY: MONSOON_MODE_ACTIVE"})
    elif scenario == 'GRIDLOCK':
        return jsonify({"traffic": 5.0, "rain": 12, "msg": "EMERGENCY: TRAFFIC_COLLAPSE"})
    else:
        return jsonify({"traffic": 1.1, "rain": 5, "msg": "SYSTEM_NOMINAL"})


# --- EXECUTION ---
if __name__ == '__main__':
    # Running on port 8000 to match your frontend fetch calls
    print("\n" + "=" * 40)
    print(" BLOOD-LINK PREDICTIVE ENGINE V4.0")
    print(" HEAL-A-THON 2026")
    print("=" * 40 + "\n")
    app.run(host='0.0.0.0', port=8000, debug=True)
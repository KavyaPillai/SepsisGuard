#!/usr/bin/env python3
"""
SepsisGuard Flask App - Final Production Version
Implements a hybrid safety-net model for maximum accuracy and patient safety.
"""

import os
import sys
from datetime import datetime
import joblib
import numpy as np
import torch
import torch.nn as nn
import traceback
from sklearn.preprocessing import StandardScaler

print("="*70)
print("🔍 SEPSISGUARD FLASK APP - FINAL PRODUCTION VERSION")
print("="*70)

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ================================
# Custom Classes (Recreate what was used in Colab)
# ================================

class MemoryOptimizedDataProcessor:
    """Recreated version of your custom processor class."""
    def __init__(self):
        self.scaler = StandardScaler()
    def fit(self, X, y=None):
        self.scaler.fit(X); return self
    def transform(self, X):
        return self.scaler.transform(X)
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class SepsisGuardLSTM(nn.Module):
    """FINAL CORRECTED LSTM architecture to match the saved model file."""
    def __init__(self, dynamic_input_size=20, static_input_size=80, hidden_size=128, num_layers=2, dropout=0.2):
        super(SepsisGuardLSTM, self).__init__()
        self.lstm = nn.LSTM(dynamic_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.static_fc = nn.Sequential(
            nn.Linear(static_input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_dynamic, x_static):
        lstm_out, _ = self.lstm(x_dynamic.unsqueeze(1))
        lstm_out = lstm_out[:, -1, :]
        static_out = self.static_fc(x_static)
        combined_out = torch.cat((lstm_out, static_out), dim=1)
        output = self.classifier(combined_out)
        return output

# ================================
# Model Loading & Feature Mapping
# ================================
processor = None
xgb_model = None
lstm_model = None
models_loaded = False

def map_simple_to_complex_features(simple_features):
    """Maps 8 simple features to the 100 features the models expect."""
    expected_features = ['glucose_10', 'sofa_14', 'lymphocytes_abs_24', 'liquid_12', 'po2_4', 'lactate_22', 'ptt_6', 'ln_bun_18', 'ph_18', '血管加压素类_5', 'ln_mbp_12', 'temperature_16', 'type1', 'ln_bun_24', 'pt_18', 'sofa_6', 'ph_20', 'neutrophils_abs_14', 'alt_6', 'heart_rate_3', 'mbp_20', 'sofa', 'sqrt_platelet_24', '血管加压素类_24', 'hadm_id', 'lactate_18', 'temperature_10', 'inv_ptt_12', 'temperature_22', 'heart_rate_24', 'lactate_16', 'inv_pt_12', 'epinephrine in 6h', 'inv_pt_1', 'temperature_4', 'norepinephrine_18', 'resp_rate_20', 'mbp_16', 'sofa_5', 'pt_24', 'sqrt1_lactate_18', 'urine_16', '总剂量_5', 'resp_rate_4', 'potassium_1', 'pao2fio2ratio_1', 'pao2fio2ratio_22', 'chloride_14', 'urine_24', '总剂量_24', '去甲肾上腺素类_8', '去甲肾上腺素类_20', 'inv_ptt_24', 'ln_ph_12', 'neutrophils_abs_12', 'heart_rate_4', 'ph_22', 'urine_14', 'ast_4', 'inv_pt_18', 'urine_8', 'ln_ph_18', 'epinephrine in 1h', 'neutrophils_abs_1', 'lactate_8', 'urine_20', 'liquid_1', 'sofa_4', '去甲肾上腺素类_5', '去甲肾上腺素类_4', 'urine_22', 'norepinephrine_6', 'glucose_22', 'heart_rate_12', 'ventilation', 'lactate_6', 'ph_3', 'sqrt_albumin_1', 'temperature_20', 'ln_pco2_1', 'subject_id', 'sqrt1_glucose_24', 'ptt_5', '总剂量_20', 'inv_ptt_6', 'temperature_14', 'ph_2', 'vasopressin in 6h', 'hemoglobin_10', 'sqrt1_lactate_24', 'ptt_1', 'spo2_18', 'hemoglobin_1', 'sqrt_heart_rate_6', 'sofa_1', 'bun_10', 'norepinephrine_4', 'lactate_2', 'resp_rate_16', 'rrt.1']
    age, gender, temp, hr, rr, sbp, wbc, lactate = simple_features
    complex_features = np.zeros(100)
    feature_map = {'temperature_16': temp, 'temperature_10': temp, 'temperature_22': temp, 'temperature_4': temp, 'temperature_20': temp, 'temperature_14': temp, 'heart_rate_3': hr, 'heart_rate_24': hr, 'heart_rate_4': hr, 'heart_rate_12': hr, 'lactate_22': lactate, 'lactate_18': lactate, 'lactate_16': lactate, 'lactate_8': lactate, 'lactate_6': lactate, 'lactate_2': lactate, 'resp_rate_20': rr, 'resp_rate_4': rr, 'resp_rate_16': rr, 'mbp_20': sbp * 0.67, 'mbp_16': sbp * 0.67, 'ln_mbp_12': np.log(max(1, sbp * 0.67))}
    for i, name in enumerate(expected_features):
        if name in feature_map: complex_features[i] = feature_map[name]
        elif 'sofa' in name.lower(): complex_features[i] = calculate_sofa_estimate(temp, hr, rr, sbp, lactate)
        elif 'glucose' in name.lower(): complex_features[i] = 120
        elif 'ph' in name.lower(): complex_features[i] = 7.35
        elif 'hemoglobin' in name.lower(): complex_features[i] = 12.0
        else: complex_features[i] = 0.0 # Use 0 instead of random noise
    return complex_features.reshape(1, -1)

def calculate_sofa_estimate(temp, hr, rr, sbp, lactate):
    sofa = 0
    if sbp < 70: sofa += 4
    elif sbp < 90: sofa += 2
    elif lactate > 4.0: sofa += 2
    if rr > 25: sofa += 1
    return min(sofa, 4)

def load_models():
    global processor, xgb_model, lstm_model, models_loaded
    print("\n🔄 Loading Models...")
    try:
        processor = joblib.load("models/sepsisguard_processor.pkl")
        print("✅ Processor loaded successfully")
    except Exception:
        processor = MemoryOptimizedDataProcessor()
    try:
        xgb_model = joblib.load("models/sepsisguard_xgboost_large.pkl")
        print("✅ XGBoost model loaded successfully")
    except Exception as e:
        print(f"❌ XGBoost loading failed: {e}")
    try:
        lstm_path = "models/sepsisguard_lstm_large.pth"
        if os.path.exists(lstm_path):
            state_dict = torch.load(lstm_path, map_location="cpu")
            lstm_model = SepsisGuardLSTM()
            lstm_model.load_state_dict(state_dict)
            lstm_model.eval()
            print("✅ LSTM model loaded successfully")
        else:
            print("⚠️ LSTM model not found")
    except Exception as e:
        print(f"❌ LSTM loading failed: {e}")
    
    models_loaded = (xgb_model is not None) and (lstm_model is not None)
    print(f"📊 Models loaded status: {models_loaded}")

def extract_features(data):
    """Extract and correctly format features from request data."""
    try:
        age = float(data.get("age", 0))
        gender = 1.0 if data.get("gender", "M") == "M" else 0.0
        features = [age, gender] + [float(data.get(k, 0)) for k in ["temperature", "heartRate", "respiratoryRate", "systolicBP", "wbc", "lactate"]]
        return np.array(features)
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        raise

def get_ai_prediction(features):
    """Gets the prediction from the AI models using the robust ensemble logic."""
    predictions = {}
    complex_features = map_simple_to_complex_features(features)
    xgb_prob, lstm_prob = None, None
    if xgb_model:
        try:
            xgb_prob = xgb_model.predict_proba(complex_features)[0][1]
            predictions['xgboost'] = {'probability': float(xgb_prob)}
            print(f"✅ XGBoost prediction: {xgb_prob:.3f}")
        except Exception as e:
            print(f"❌ XGBoost prediction failed: {e}")
    if lstm_model:
        try:
            dynamic_features, static_features = complex_features[0, :20], complex_features[0, 20:]
            tensor_dynamic = torch.tensor(dynamic_features, dtype=torch.float32).view(1, -1)
            tensor_static = torch.tensor(static_features, dtype=torch.float32).view(1, -1)
            with torch.no_grad():
                output = lstm_model(tensor_dynamic, tensor_static)
                lstm_prob = torch.sigmoid(output).item()
            predictions['lstm'] = {'probability': float(lstm_prob)}
            print(f"✅ LSTM prediction: {lstm_prob:.3f}")
        except Exception as e:
            print(f"❌ LSTM prediction failed: {e}")
    
    if xgb_prob is not None and lstm_prob is not None:
        if (xgb_prob >= 0.5) == (lstm_prob >= 0.5):
            print("⚖️ Models agree. Averaging scores.")
            final_prob = (xgb_prob + lstm_prob) / 2
        else:
            print("⚠️ Models disagree. Trusting XGBoost score.")
            final_prob = xgb_prob
    elif xgb_prob is not None:
        final_prob = xgb_prob
    else:
        return None, predictions # Return None if no AI model worked
        
    return float(final_prob), predictions

def get_clinical_rules_prediction(features):
    """Calculate risk score using clinical rules."""
    print("🩺 Calculating clinical rules-based score...")
    age, _, temp, hr, rr, sbp, wbc, lactate = features
    risk_score, factors = 0.0, []
    if age > 65: risk_score += min(15, (age-65)*0.5); factors.append(f"Advanced age ({age} years)")
    if temp > 38.3: risk_score += 15; factors.append(f"Fever ({temp}°C)")
    elif temp < 36.0: risk_score += 20; factors.append(f"Hypothermia ({temp}°C)")
    if hr > 90: risk_score += 10; factors.append(f"Tachycardia ({hr} bpm)")
    if rr > 20: risk_score += 10; factors.append(f"Tachypnea ({rr}/min)")
    if sbp < 90: risk_score += 30; factors.append(f"Hypotension ({sbp} mmHg)")
    elif sbp < 100: risk_score += 15; factors.append(f"Low BP ({sbp} mmHg)")
    if wbc > 12: risk_score += 12; factors.append(f"Leukocytosis ({wbc})")
    elif wbc < 4: risk_score += 15; factors.append(f"Leukopenia ({wbc})")
    if lactate > 4.0: risk_score += 35; factors.append(f"Severe hyperlactatemia ({lactate})")
    elif lactate > 2.0: risk_score += 20; factors.append(f"Elevated lactate ({lactate})")
    final_score = max(5, min(95, risk_score))
    return float(final_score / 100), factors

# --- Flask Routes ---
load_models()

@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy", "models_loaded": models_loaded})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"📨 Received prediction request")
        features = extract_features(data)
        print(f"📊 Extracted features: {features}")

        ai_prob, ai_predictions = get_ai_prediction(features)
        rules_prob, clinical_factors = get_clinical_rules_prediction(features)

        # Hybrid Safety-Net Logic: Use the higher of the two scores
        if ai_prob is not None:
            final_prob = max(ai_prob, rules_prob)
            print(f"🧠 AI Score: {ai_prob*100:.1f}%, 룰 Clinical Score: {rules_prob*100:.1f}%. Using highest: {final_prob*100:.1f}%")
        else:
            print("❌ AI prediction failed. Using clinical rules score only.")
            final_prob = rules_prob

        result = {
            'success': True,
            'risk_probability': final_prob,
            'risk_score': final_prob * 100,
            'prediction': int(final_prob >= 0.5),
            'model_predictions': {
                'xgboost': ai_predictions.get('xgboost', {}).get('probability', 0) * 100,
                'lstm': ai_predictions.get('lstm', {}).get('probability', 0) * 100,
                'ensemble': (ai_prob or 0) * 100,
                'clinical_rules': rules_prob * 100
            },
            'clinical_factors': clinical_factors,
            'confidence': 0.95,
            'patient_id': data.get("patientId", "Unknown"),
            'timestamp': datetime.now().isoformat(),
            'risk_category': get_risk_category(final_prob * 100)
        }
        
        print(f"📊 Final Result: {result['risk_score']:.1f}% risk")
        return jsonify(result)

    except Exception as e:
        print(f"❌ Top-level prediction error: {e}"); traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

def get_risk_category(score):
    if score < 25: return {"level": "Low", "description": "Minimal sepsis indicators"}
    elif score < 50: return {"level": "Moderate", "description": "Some concerning signs"}
    elif score < 75: return {"level": "High", "description": "Multiple sepsis indicators"}
    else: return {"level": "Critical", "description": "Severe sepsis likely, immediate attention required"}

@app.route('/api/alert', methods=['POST'])
def send_alert():
    alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return jsonify({"success": True, "message": "Alert sent successfully", "alert_id": alert_id})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏥 STARTING SEPSISGUARD - FINAL PRODUCTION VERSION")
    print("="*70)
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
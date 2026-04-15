#src/api/main.py
import os
import sys
import json
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.insert(0, PROJECT_ROOT)

app = FastAPI(title="Neon-Red Manufacturing Intelligence")

# Allow Frontend to talk to Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
INTEL_PATH = "models/system_intelligence.json"

@app.post("/system/audit")
async def perform_full_audit(sensor_input: dict):
    """
    ONE INPUT -> FULL INTELLIGENCE
    The dashboard sends all slider values here.
    """
    try:
        # 1. Load Models from MLflow Registry
        # Logic: Separation of training and serving
        rul_model = mlflow.sklearn.load_model("models:/NASA_RUL_Predictor/latest")
        anomaly_model = mlflow.sklearn.load_model("models:/AI4I_Anomaly_Detector/latest")
        
        # 2. Unified Feature Engineering (The 'Physics' Layer)
        df = pd.DataFrame([sensor_input])
        
        # NASA Physics
        df['thermal_stress'] = df.get('s11', 0) * df.get('s15', 0)
        df['pressure_heat_index'] = df.get('s4', 0) * df.get('s11', 0)
        
        # AI4I Physics
        df['temp_delta'] = df.get('proc_temp_c', 0) - df.get('air_temp_c', 0)
        df['power_w'] = df.get('torque', 0) * (df.get('rpm', 0) * 0.1047)
        df['stress_index'] = (df.get('torque', 0) / 75.0) * (df.get('rpm', 0) / 2800.0)

        # 3. Dual-Core Prediction
        # Column selection happens here to ensure model compatibility
        nasa_cols = ['thermal_stress', 'pressure_heat_index', 's4_velocity', 's4_trend', 's11_velocity', 's11_trend', 's15_velocity', 's15_trend']
        ai4i_cols = ["air_temp_c", "proc_temp_c", "rpm", "torque", "tool_wear", "power_w", "stress_index", "temp_delta"]
        
        # Handle missing columns for "Cold Start" safety
        for c in nasa_cols + ai4i_cols:
            if c not in df.columns: df[c] = 0.0

        rul_val = rul_model.predict(df[nasa_cols])[0]
        anomaly_val = anomaly_model.predict(df[ai4i_cols])[0] # -1 is anomaly

        # 4. Load Causal Truths
        with open(INTEL_PATH, "r") as f:
            intel = json.load(f)

        # 5. THE OUTPUT: Designed for the Neon Dashboard
        return {
            "status": "CRITICAL" if rul_val < 30 or anomaly_val == -1 else "STABLE",
            "prognostics": {
                "predicted_rul": round(float(rul_val), 2),
                "life_bar_percent": max(0, min(100, (rul_val / 200) * 100))
            },
            "diagnostics": {
                "anomaly_detected": True if anomaly_val == -1 else False,
                "health_score": round(float(anomaly_model.decision_function(df[ai4i_cols])[0]), 4),
                "power_consumption_w": round(df['power_w'].iloc[0], 2)
            },
            "causal_forensics": {
                "key_driver": "Thermal Stress",
                "impact_coefficient": intel.get("nasa_insights", {}).get("thermal_stress_impact", -23.78),
                "recommendation": "Reduce Torque by 15% to stabilize RUL"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    # This keeps the script alive and listening on port 8000
    print("🚀 NEON-RED BACKEND STARTING ON http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI
from pydantic import BaseModel
from gradio_client import Client as GradioClient
from groq import Groq
import numpy as np
import json

app = FastAPI(title="Wind Turbine Optimization API", version="2.4")

# Initialize clients
gr_client = GradioClient("PradeepKumar11519/scenario-prediction-model")
groq_client = Groq(api_key="YOUR_GROQ_API_KEY_HERE")


# ---------------------------
# Input schema
# ---------------------------
class InputData(BaseModel):
    WindSpeed: float
    StdDevWindSpeed: float
    WindDirAbs: float
    WindDirRel: float
    Power: float
    MaxPower: float
    MinPower: float
    StdDevPower: float
    AvgRPow: float
    Pitch: float
    GenRPM: float
    RotorRPM: float
    EnvirTemp: float
    NacelTemp: float
    GearOilTemp: float
    GearBearTemp: float
    GenTemp: float
    GenPh1Temp: float
    GenPh2Temp: float
    GenPh3Temp: float
    GenBearTemp: float
    Yaw: float


# ---------------------------
# Physics-based model
# ---------------------------
def calculate_power(wind_speed: float, pitch: float, rpm: float) -> float:
    """Estimate turbine power using simplified aerodynamic model."""
    air_density = 1.225
    blade_radius = 40
    swept_area = np.pi * blade_radius ** 2
    rated_power = 2500

    if wind_speed <= 0:
        return 0

    pitch_rad = np.radians(pitch)
    tip_speed = (rpm * 2 * np.pi * blade_radius) / 60
    tsr = tip_speed / wind_speed

    tsr_opt = 7.5
    tsr_eff = np.exp(-((tsr - tsr_opt) / 3) ** 2)
    pitch_eff = np.cos(pitch_rad) * (1 - abs(pitch_rad) / (np.pi / 2))
    cp = 0.45 * pitch_eff * tsr_eff

    theoretical = 0.5 * air_density * swept_area * (wind_speed ** 3) * cp / 1000
    return min(max(theoretical, 0), rated_power)


# ---------------------------
# Brute-force optimization
# ---------------------------
def optimize_turbine(wind_speed: float, wind_direction: float, pitch: float, rpm: float, yaw: float):
    best_power = 0
    best_pitch = pitch
    best_rpm = rpm

    pitch_range = np.arange(0, 25, 0.5)
    rpm_range = np.arange(10, 20, 0.5)

    for p in pitch_range:
        for r in rpm_range:
            power = calculate_power(wind_speed, p, r)
            if power > best_power:
                best_power = power
                best_pitch = p
                best_rpm = r

    return {
        "power": best_power,
        "pitch": best_pitch,
        "rpm": best_rpm,
        "yaw": wind_direction
    }


# ---------------------------
# Structural Stress Model
# ---------------------------
def calculate_structural_stress(wind_speed: float, rpm: float, pitch: float, temp: float):
    """
    Calculate detailed structural stress breakdown:
    1) Blade Load      ∝ wind_speed² × (1 - pitch/25)
    2) Tower Stress    ∝ wind_speed × rpm / 20
    3) Fatigue         ∝ rpm³ / 10000
    4) Gearbox Heat    ∝ temp / 100 × rpm / 20
    Normalized to [0,1] and combined as weighted average.
    """
    blade_load = np.clip((wind_speed ** 2) * (1 - pitch / 25) / 500, 0, 1)
    tower_stress = np.clip((wind_speed * rpm / 20) / 500, 0, 1)
    fatigue = np.clip((rpm ** 3) / 10000 / 1000, 0, 1)
    gearbox_heat = np.clip((temp / 100) * (rpm / 20), 0, 1)

    total_stress = 0.4 * blade_load + 0.3 * tower_stress + 0.2 * fatigue + 0.1 * gearbox_heat
    return {
        "blade_load": round(blade_load, 3),
        "tower_stress": round(tower_stress, 3),
        "fatigue": round(fatigue, 3),
        "gearbox_heat": round(gearbox_heat, 3),
        "total_stress": round(total_stress, 3)
    }


# ---------------------------
# Main endpoint
# ---------------------------
@app.post("/predict")
def predict_endpoint(data: InputData):
    # Step 1: Current performance
    current_power = calculate_power(data.WindSpeed, data.Pitch, data.GenRPM)
    current_stress = calculate_structural_stress(data.WindSpeed, data.GenRPM, data.Pitch, data.GenTemp)

    # Step 2: Optimized performance
    optimized = optimize_turbine(
        wind_speed=data.WindSpeed,
        wind_direction=data.WindDirAbs,
        pitch=data.Pitch,
        rpm=data.GenRPM,
        yaw=data.Yaw
    )
    optimized_stress = calculate_structural_stress(data.WindSpeed, optimized["rpm"], optimized["pitch"], data.GenTemp)

    # Step 3: Compute improvements
    power_improvement_kw = optimized["power"] - current_power
    power_improvement_percent = (power_improvement_kw / current_power * 100) if current_power > 0 else 0
    annual_energy_increase_mwh = (power_improvement_kw * 8760) / 1000
    estimated_revenue_increase = annual_energy_increase_mwh * 50  # assume $50/MWh
    structural_stress_reduction = (current_stress["total_stress"] - optimized_stress["total_stress"]) * 100

    # Step 4: Optimization recommendations
    optimization_steps = []
    if abs(data.Pitch - optimized["pitch"]) > 0.5:
        direction = "Decrease" if data.Pitch > optimized["pitch"] else "Increase"
        optimization_steps.append(f"Step 1: {direction} blade pitch from {data.Pitch:.1f}° → {optimized['pitch']:.1f}°")
    if abs(data.GenRPM - optimized["rpm"]) > 0.5:
        direction = "Decrease" if data.GenRPM > optimized["rpm"] else "Increase"
        optimization_steps.append(f"Step 2: {direction} generator RPM from {data.GenRPM:.1f} → {optimized['rpm']:.1f}")
    if abs(data.Yaw - optimized["yaw"]) > 5:
        optimization_steps.append(f"Step 3: Adjust yaw alignment from {data.Yaw:.0f}° → {optimized['yaw']:.0f}°")
    if not optimization_steps:
        optimization_steps.append("Turbine is already operating near optimal settings.")

    # Step 5: ML model prediction
    try:
        ml_output = gr_client.predict(
            param_0=data.WindSpeed,
            param_1=data.StdDevWindSpeed,
            param_2=data.WindDirAbs,
            param_3=data.WindDirRel,
            param_4=data.Power,
            param_5=data.MaxPower,
            param_6=data.MinPower,
            param_7=data.StdDevPower,
            param_8=data.AvgRPow,
            param_9=data.Pitch,
            param_10=data.GenRPM,
            param_11=data.RotorRPM,
            param_12=data.EnvirTemp,
            param_13=data.NacelTemp,
            param_14=data.GearOilTemp,
            param_15=data.GearBearTemp,
            param_16=data.GenTemp,
            param_17=data.GenPh1Temp,
            param_18=data.GenPh2Temp,
            param_19=data.GenPh3Temp,
            param_20=data.GenBearTemp,
            api_name="/predict_manual"
        )
        if isinstance(ml_output, str):
            ml_output = json.loads(ml_output)
    except Exception as e:
        ml_output = {"error": f"Gradio call failed: {str(e)}"}

    # Step 6: AI summary
    prompt = f"""
Wind turbine optimization summary:
- Wind Speed: {data.WindSpeed:.2f} m/s
- Current Power: {current_power:.2f} kW
- Optimized Power: {optimized['power']:.2f} kW
- Power Improvement: {power_improvement_percent:.2f}%
- Structural Stress Reduction: {structural_stress_reduction:.2f}%
- ML Prediction: {ml_output.get('Predicted Class', 'Unknown')}
- Optimal Pitch: {optimized['pitch']:.2f}°, RPM: {optimized['rpm']:.2f}

Summarize in 2–3 lines with recommended actions and stress impact.
"""
    try:
        groq_response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_completion_tokens=150,
            top_p=1,
            reasoning_effort="medium"
        )
        summary = groq_response.choices[0].message.content.strip()
    except Exception as e:
        summary = (
            f"Optimization suggests a {power_improvement_percent:.1f}% power increase "
            f"and {structural_stress_reduction:.1f}% stress reduction."
        )

    # Step 7: Return structured response
    return {
        "summary": summary,
        "telemetry": {
            "wind_speed": data.WindSpeed,
            "pitch": data.Pitch,
            "rpm": data.GenRPM,
            "yaw": data.Yaw,
            "environment": {
                "temperature": data.EnvirTemp,
                "nacelle_temp": data.NacelTemp,
                "gear_oil_temp": data.GearOilTemp,
                "generator_temp": data.GenTemp
            }
        },
        "current_state": {
            "power_kw": round(current_power, 2),
            "stress_breakdown": current_stress
        },
        "optimized_state": {
            "power_kw": round(optimized["power"], 2),
            "pitch_deg": round(optimized["pitch"], 2),
            "rpm": round(optimized["rpm"], 2),
            "yaw_deg": round(optimized["yaw"], 2)
        },
        "performance_metrics": {
            "power_improvement_percent": round(power_improvement_percent, 2),
            "power_gain_kw": round(power_improvement_kw, 2),
            "annual_energy_gain_mwh": round(annual_energy_increase_mwh, 2),
            "estimated_revenue_gain_usd": round(estimated_revenue_increase, 2),
            "structural_stress_reduction_percent": round(structural_stress_reduction, 2)
        },
        "ml_prediction": ml_output,
        "optimization_steps": optimization_steps
    }


# ---------------------------
# Root endpoint
# ---------------------------
@app.get("/")
def root():
    return {
        "message": "Welcome to Wind Turbine Optimization API",
        "version": "2.4",
        "endpoints": {
            "/predict": "POST - Submit turbine telemetry to receive optimization report",
            "/docs": "Interactive API documentation"
        }
    }


# ---------------------------
# Local dev entry point
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

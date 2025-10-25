from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from gradio_client import Client as GradioClient
from groq import Groq
import numpy as np
import pandas as pd
import json
from io import StringIO
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import asyncio
import os

app = FastAPI(title="Wind Turbine Optimization API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize clients
gr_client = GradioClient("PradeepKumar11519/scenario-prediction-model")
groq_client = Groq(api_key="YOUR_GROQ_API_KEY_HERE")

# Predefined CSV file path
PREDEFINED_CSV = "./test_wtih_ts.csv"

# ---------------------------
# Required CSV columns
# ---------------------------
REQUIRED_COLUMNS = [
    'Timestamp',
    'WindSpeed',
    'StdDevWindSpeed',
    'WindDirAbs',
    'WindDirRel',
    'Power',
    'MaxPower',
    'MinPower',
    'StdDevPower',
    'AvgRPow',
    'Pitch',
    'GenRPM',
    'RotorRPM',
    'EnvirTemp',
    'NacelTemp',
    'GearOilTemp',
    'GearBearTemp',
    'GenTemp',
    'GenPh1Temp',
    'GenPh2Temp',
    'GenPh3Temp',
    'GenBearTemp',
    'Yaw'
]


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
    """Calculate detailed structural stress breakdown."""
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
# Process single row
# ---------------------------
def process_turbine_row(row):
    """Process a single turbine data row and return optimization results."""
    # Step 1: Current performance
    current_power = calculate_power(row['WindSpeed'], row['Pitch'], row['GenRPM'])
    current_stress = calculate_structural_stress(row['WindSpeed'], row['GenRPM'], row['Pitch'], row['GenTemp'])

    # Step 2: Optimized performance
    optimized = optimize_turbine(
        wind_speed=row['WindSpeed'],
        wind_direction=row['WindDirAbs'],
        pitch=row['Pitch'],
        rpm=row['GenRPM'],
        yaw=row['Yaw']
    )
    optimized_stress = calculate_structural_stress(row['WindSpeed'], optimized["rpm"], optimized["pitch"], row['GenTemp'])

    # Step 3: Compute improvements
    power_improvement_kw = optimized["power"] - current_power
    power_improvement_percent = (power_improvement_kw / current_power * 100) if current_power > 0 else 0
    annual_energy_increase_mwh = (power_improvement_kw * 8760) / 1000
    estimated_revenue_increase = annual_energy_increase_mwh * 50
    structural_stress_reduction = (current_stress["total_stress"] - optimized_stress["total_stress"]) * 100

    # Step 4: Optimization recommendations
    optimization_steps = []
    if abs(row['Pitch'] - optimized["pitch"]) > 0.5:
        direction = "Decrease" if row['Pitch'] > optimized["pitch"] else "Increase"
        optimization_steps.append(f"Step 1: {direction} blade pitch from {row['Pitch']:.1f}° → {optimized['pitch']:.1f}°")
    if abs(row['GenRPM'] - optimized["rpm"]) > 0.5:
        direction = "Decrease" if row['GenRPM'] > optimized["rpm"] else "Increase"
        optimization_steps.append(f"Step 2: {direction} generator RPM from {row['GenRPM']:.1f} → {optimized['rpm']:.1f}")
    if abs(row['Yaw'] - optimized["yaw"]) > 5:
        optimization_steps.append(f"Step 3: Adjust yaw alignment from {row['Yaw']:.0f}° → {optimized['yaw']:.0f}°")
    if not optimization_steps:
        optimization_steps.append("Turbine is already operating near optimal settings.")

    # Step 5: ML model prediction
    try:
        ml_output = gr_client.predict(
            param_0=row['WindSpeed'],
            param_1=row['StdDevWindSpeed'],
            param_2=row['WindDirAbs'],
            param_3=row['WindDirRel'],
            param_4=row['Power'],
            param_5=row['MaxPower'],
            param_6=row['MinPower'],
            param_7=row['StdDevPower'],
            param_8=row['AvgRPow'],
            param_9=row['Pitch'],
            param_10=row['GenRPM'],
            param_11=row['RotorRPM'],
            param_12=row['EnvirTemp'],
            param_13=row['NacelTemp'],
            param_14=row['GearOilTemp'],
            param_15=row['GearBearTemp'],
            param_16=row['GenTemp'],
            param_17=row['GenPh1Temp'],
            param_18=row['GenPh2Temp'],
            param_19=row['GenPh3Temp'],
            param_20=row['GenBearTemp'],
            api_name="/predict_manual"
        )
        if isinstance(ml_output, str):
            ml_output = json.loads(ml_output)
    except Exception as e:
        ml_output = {"error": f"Gradio call failed: {str(e)}"}

    # Step 6: AI summary
    prompt = f"""
Wind turbine optimization summary:
- Wind Speed: {row['WindSpeed']:.2f} m/s
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

    # Return structured response
    return {
        "timestamp": str(row['Timestamp']),
        "summary": summary,
        "telemetry": {
            "wind_speed": row['WindSpeed'],
            "pitch": row['Pitch'],
            "rpm": row['GenRPM'],
            "yaw": row['Yaw'],
            "environment": {
                "temperature": row['EnvirTemp'],
                "nacelle_temp": row['NacelTemp'],
                "gear_oil_temp": row['GearOilTemp'],
                "generator_temp": row['GenTemp']
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
# Helper function to load CSV
# ---------------------------
def load_predefined_csv():
    """Load the predefined CSV file."""
    if not os.path.exists(PREDEFINED_CSV):
        raise FileNotFoundError(f"CSV file '{PREDEFINED_CSV}' not found in current directory")
    
    df = pd.read_csv(PREDEFINED_CSV)
    
    # Validate required columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    return df


# ---------------------------
# NEW: Process Predefined CSV (All at once)
# ---------------------------
@app.get("/predict-all")
async def predict_all():
    """
    Process ALL rows from the predefined CSV file (test_wtih_ts.csv).
    Returns complete results for all rows.
    """
    try:
        df = load_predefined_csv()
        
        results = []
        for idx, row in df.iterrows():
            try:
                result = process_turbine_row(row)
                results.append(result)
            except Exception as e:
                results.append({
                    "timestamp": str(row.get('Timestamp', 'Unknown')),
                    "error": f"Processing failed: {str(e)}"
                })
        
        return {
            "status": "success",
            "csv_file": PREDEFINED_CSV,
            "total_records": len(df),
            "processed_records": len(results),
            "results": results
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.get("/predict-stream/{limit}")
async def predict_stream(limit: int = None, include_total_rows: bool = False):
    """
    Stream predictions row-by-row from predefined CSV.
    Optional: specify /predict-stream/{limit} to process only top N rows.
    Example:
      /predict-stream            → stream all rows
      /predict-stream/20         → stream only first 20 rows
      /predict-stream/20?include_total_rows=true
    """
    async def generate_stream():
        try:
            df = load_predefined_csv()
            total_rows = len(df)

            # Trim to top N rows if limit provided
            if limit and limit > 0:
                df = df.head(limit)
                message = f"Processing top {limit} rows (out of {total_rows})..."
            else:
                message = f"Processing all {total_rows} rows..."

            selected_count = len(df)

            # Initial metadata message
            yield json.dumps({
                "type": "metadata",
                "csv_file": PREDEFINED_CSV,
                "total_rows": total_rows,
                "selected_rows": selected_count,
                "message": message
            }) + "\n"

            # Optional total count message
            if include_total_rows:
                yield json.dumps({
                    "type": "total_rows",
                    "count": selected_count,
                    "message": f"Total rows being processed: {selected_count}"
                }) + "\n"

            # Process and stream each row
            for idx, row in df.iterrows():
                try:
                    result = process_turbine_row(row)
                    yield json.dumps({
                        "type": "row_result",
                        "row_number": idx + 1,
                        "total_rows": selected_count,
                        "progress_percent": round((idx + 1) / selected_count * 100, 2),
                        "data": result
                    }) + "\n"
                except Exception as e:
                    yield json.dumps({
                        "type": "row_error",
                        "row_number": idx + 1,
                        "total_rows": selected_count,
                        "timestamp": str(row.get('Timestamp', 'Unknown')),
                        "error": str(e)
                    }) + "\n"

                await asyncio.sleep(0.1)

            # Completion message
            yield json.dumps({
                "type": "complete",
                "total_rows": selected_count,
                "message": "Processing complete"
            }) + "\n"

        except FileNotFoundError as e:
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "error": f"Error processing CSV: {str(e)}"}) + "\n"

    return StreamingResponse(
        generate_stream(),
        media_type="application/x-ndjson"
    )




# ---------------------------
# NEW: Get specific row from predefined CSV
# ---------------------------
@app.get("/predict-row/{row_number}")
async def predict_specific_row(row_number: int):
    """
    Process a specific row number from the predefined CSV file.
    Row numbers start from 1.
    """
    try:
        df = load_predefined_csv()
        total_rows = len(df)
        
        if row_number < 1 or row_number > total_rows:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid row number. Must be between 1 and {total_rows}"
            )
        
        # Get the specific row (convert to 0-indexed)
        row = df.iloc[row_number - 1]
        result = process_turbine_row(row)
        
        return {
            "status": "success",
            "csv_file": PREDEFINED_CSV,
            "row_number": row_number,
            "total_rows": total_rows,
            "result": result
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ---------------------------
# NEW: Get CSV info
# ---------------------------
@app.get("/csv-info")
async def get_csv_info():
    """
    Get information about the predefined CSV file.
    """
    try:
        df = load_predefined_csv()
        
        return {
            "csv_file": PREDEFINED_CSV,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "first_timestamp": str(df.iloc[0]['Timestamp']) if len(df) > 0 else None,
            "last_timestamp": str(df.iloc[-1]['Timestamp']) if len(df) > 0 else None,
            "sample_data": df.head(3).to_dict('records')
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV: {str(e)}")


# ---------------------------
# OPTIONAL: Original CSV Upload endpoint
# ---------------------------
@app.post("/predict")
async def predict_csv_endpoint(file: UploadFile = File(...)):
    """
    Upload a CSV file with turbine telemetry data.
    Returns optimization results for ALL rows at once.
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    
    try:
        # Read CSV file
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data)
        
        # Validate required columns
        missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Process each row
        results = []
        for idx, row in df.iterrows():
            try:
                result = process_turbine_row(row)
                results.append(result)
            except Exception as e:
                results.append({
                    "timestamp": str(row.get('Timestamp', 'Unknown')),
                    "error": f"Processing failed: {str(e)}"
                })
        
        return {
            "status": "success",
            "total_records": len(df),
            "processed_records": len(results),
            "results": results
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")


# ---------------------------
# Root endpoint
# ---------------------------
@app.get("/")
def root():
    return {
        "message": "Welcome to Wind Turbine Optimization API",
        "version": "3.0",
        "predefined_csv": PREDEFINED_CSV,
        "endpoints": {
            "/csv-info": "GET - Get info about the predefined CSV file",
            "/predict-all": "GET - Process ALL rows from predefined CSV at once",
            "/predict-stream": "GET - Process predefined CSV row-by-row with streaming",
            "/predict-row/{row_number}": "GET - Process a specific row number (1-indexed)",
            "/predict": "POST - Upload and process a custom CSV file",
            "/health": "GET - Health check",
            "/docs": "GET - Interactive API documentation"
        },
        "usage_examples": {
            "get_csv_info": "GET /csv-info",
            "process_all_rows": "GET /predict-all",
            "stream_processing": "GET /predict-stream",
            "specific_row": "GET /predict-row/5"
        }
    }


# ---------------------------
# Health check endpoint
# ---------------------------
@app.get("/health")
def health_check():
    csv_exists = os.path.exists(PREDEFINED_CSV)
    return {
        "status": "healthy",
        "version": "3.0",
        "csv_file": PREDEFINED_CSV,
        "csv_exists": csv_exists
    }


# ---------------------------
# Local dev entry point
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
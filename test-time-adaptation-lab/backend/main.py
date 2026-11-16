from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
import io
from engine import ExperimentEngine

app = FastAPI(title="TTA Lab API", version="1.0.0",
              description="Interactive Test-Time Adaptation research demonstrator.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
engine = ExperimentEngine()
LAST_EXPERIMENT = None

class ExperimentConfig(BaseModel):
    noise_sigma: float = Field(0.30, ge=0, le=1)
    batch_size: int = Field(128, ge=8, le=512)
    tent_learning_rate: float = Field(0.0001, ge=0.000001, le=0.01)
    noise_seed: int = Field(43, ge=0, le=999999)
    confidence_gate: bool = False

class TrainConfig(BaseModel):
    epochs: int = Field(8, ge=1, le=50)
    learning_rate: float = Field(0.003, ge=0.00001, le=0.1)
    batch_size: int = Field(128, ge=16, le=512)
    seed: int = Field(42, ge=0, le=999999)

@app.get("/api/health")
def health():
    return {"status": "ok", "project": "Test-Time Adaptation Lab"}

@app.get("/api/overview")
def overview():
    return {"project": "Test-Time Adaptation in Deep Learning",
            "dataset": {"name": "scikit-learn Digits", "images": 1797,
                        "train": 1347, "test": 450, "classes": 10,
                        "input_shape": "3 × 32 × 32"},
            "model": engine.metadata, "training_history": engine.history,
            "parameters": __import__("model").parameter_summary(engine.model),
            "device": "CPU"}

@app.post("/api/experiment")
def experiment(config: ExperimentConfig):
    global LAST_EXPERIMENT
    try:
        LAST_EXPERIMENT = engine.run_experiment(
            sigma=config.noise_sigma, batch_size=config.batch_size,
            tent_lr=config.tent_learning_rate, seed=config.noise_seed,
            confidence_gate=config.confidence_gate)
        return LAST_EXPERIMENT
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/api/train")
def train(config: TrainConfig):
    try:
        return engine.retrain(config.epochs, config.learning_rate,
                              config.batch_size, config.seed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg", "image/webp", "image/bmp"):
        raise HTTPException(status_code=415, detail="Upload a PNG, JPG, WEBP or BMP image.")
    data = await file.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be smaller than 8 MB.")
    try:
        return engine.predict_upload(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}")

@app.get("/api/export")
def export():
    if LAST_EXPERIMENT is None:
        raise HTTPException(status_code=404, detail="Run an experiment before exporting.")
    from engine import ExperimentEngine
    csv_text = engine.export_csv(LAST_EXPERIMENT)
    return StreamingResponse(io.BytesIO(csv_text.encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=tta_experiment_results.csv"})

@app.get("/api/methods")
def methods():
    return {"methods": [
        {"name":"Source-only", "signal":"Frozen model; source BatchNorm running statistics",
         "updated":"None", "status":"Baseline"},
        {"name":"BatchNorm statistics only", "signal":"Target-batch mean and variance",
         "updated":"Statistics only; no gradients", "status":"Ablation included in this app"},
        {"name":"TENT", "signal":"Entropy minimisation on unlabelled target batches",
         "updated":"BatchNorm γ and β only", "status":"Implemented"},
        {"name":"Test-Time Training (TTT)", "signal":"Self-supervised auxiliary task",
         "updated":"Shared feature extractor", "status":"Theory only; not implemented"},
        {"name":"Test-time prompt tuning", "signal":"Entropy over augmented views",
         "updated":"Prompt tokens", "status":"Theory only; not implemented"}
    ]}

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")

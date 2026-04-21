import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from ..code.readmission.data_processing import PredictionModel
from .models import PatientInput
from .inventory import router as inventory_router
from .inventory_model_api import router as inventory_model_router
from .database import init_db

app = FastAPI()
init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Starting")

_prediction_model: PredictionModel | None = None
_model_ready = threading.Event()


def _train_readmission_model() -> None:
    global _prediction_model
    try:
        _prediction_model = PredictionModel.load_or_train()
        print("Readmission model ready")
    except Exception as e:
        print(f"Readmission model training failed: {e}")
    finally:
        _model_ready.set()


@app.on_event("startup")
def _kick_off_training() -> None:
    threading.Thread(target=_train_readmission_model, daemon=True).start()


def get_prediction_model() -> PredictionModel:
    _model_ready.wait()
    if _prediction_model is None:
        raise HTTPException(status_code=503, detail="Readmission model failed to load.")
    return _prediction_model


@app.get("/api/fields")
def get_fields():
    return PatientInput.get_fields()


@app.post("/api/predict")
async def predict(patient_data: PatientInput):
    return get_prediction_model().predict(patient_data)

@app.post("/api/predict/med-impact")
async def predict_med_impact(patient_data: PatientInput):
    return get_prediction_model().predict_med_impact(patient_data)

@app.get("/api/model_stats")
async def get_model_stats():
    return get_prediction_model().get_model_stats()


app.include_router(inventory_router)
app.include_router(inventory_model_router)

app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
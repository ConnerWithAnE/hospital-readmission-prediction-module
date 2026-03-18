from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from ..code.data_processing import PredictionModel
from .models import PatientInput

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Starting")

prediction_model = PredictionModel.load_or_train()


@app.get("/api/fields")
def get_fields():
    return PatientInput.get_fields()


@app.post("/api/predict")
async def predict(patient_data: PatientInput):
    return prediction_model.predict(patient_data)

@app.get("/api/model_stats")
async def get_model_stats():
    return prediction_model.get_model_stats()


app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
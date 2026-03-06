from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return

@app.get("/")
def read_root():
    return

@app.post("/api/predict")
async def predict(patient_data: dict):
    # TODO features = preprocess(patient_data)
    # TODO risk_score = model.predict_proba(features)[:,1][0]
    return

app.mount("/", '''StaticFiles(directory="frontend/dist", html=True)''', name="static")




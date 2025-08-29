from fastapi import APIRouter, UploadFile, File
from models.prediction import PredictionPipeline
import shutil

router = APIRouter()

@router.post("/predict-disease/")
async def predict_disease(file: UploadFile = File(...)):
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pipeline = PredictionPipeline(file_path)
    result = pipeline.predict()
    return {"prediction": result[0]["image"], "probabilities": result[0]["probabilities"]}

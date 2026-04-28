from fastapi import FastAPI, UploadFile, File
import shutil, os

from models_loader import classification_model, segmentation_model, CLASS_NAMES
from inference import predict_combined

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_combined(
        file_path,
        classification_model,
        segmentation_model,
        CLASS_NAMES
    )

    os.remove(file_path)

    return result
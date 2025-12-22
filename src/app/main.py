from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from src.pred.model_loader import predict_from_bytes

router = APIRouter()

class PredictionResponse(BaseModel):
    label: str
    confidence: float

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        label, confidence = predict_from_bytes(contents)
        return {"label": label, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from fastapi import APIRouter, UploadFile, File
import time
from app.utils.preprocessing import preprocess_image
from app.utils.postprocessing import postprocess
from app.services.triton_client import infer

router = APIRouter()


@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    start_time = time.time()

    image_array = preprocess_image(image.file)
    output = infer(image_array)
    predicted_class, confidence = postprocess(output)

    latency = (time.time() - start_time) * 1000

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "latency_ms": latency,
    }


@router.get("/health")
async def health():
    return {"status": "healthy"}

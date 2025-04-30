import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

# Load ONNX model
onnx_path = "vit_model.onnx"
session = ort.InferenceSession(onnx_path)

# FastAPI app
app = FastAPI()

# Class labels (placeholder — replace with your actual class names if available)
CLASS_NAMES = [f"Class {i}" for i in range(1000)]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).numpy()

        inputs = {session.get_inputs()[0].name: input_tensor}
        outputs = session.run(None, inputs)

        predicted_idx = int(np.argmax(outputs[0]))
        predicted_label = CLASS_NAMES[predicted_idx]

        return JSONResponse(content={"class": predicted_label})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

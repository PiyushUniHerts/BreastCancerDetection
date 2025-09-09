import numpy as np
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
import uvicorn
import io
import os 
from PIL import Image

# ==============================
# Init FastAPI
# ==============================
app = FastAPI()

# ==============================
# Load model at startup
# ==============================
MODEL_PATH = "best_densenet121.keras"   # or .h5 if using weights
model = load_model(MODEL_PATH)
input_shape = model.input_shape[1:3]  # (height, width)
print(f"✅ Model loaded. Input size = {input_shape}")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==============================
# Helper: preprocess image
# ==============================
def preprocess_img(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((50,50))  # resize to model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1,h,w,3)
    img_array = preprocess_input(img_array)
    return img_array

# ==============================
# Routes
# ==============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = preprocess_img(contents).astype("float32")
    prediction = model.predict(img_array)

    prob = float(prediction[0][0])
    label = 1 if prob > 0.5 else 0

    return {
        "probability": prob,
        "class": label
    }

# ==============================
# Run app
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT",8000))
    uvicorn.run(app,host="0.0.0.0",port=port)

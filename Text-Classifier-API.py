from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Run on command uvicorn serv:app --host {ip} --port {port} --reload

model = tf.keras.models.load_model("emotion_classifier_model_rev2.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

emotion_mapping = {
    0: "Anger",
    1: "Fear",
    2: "Joy",
    3: "Love",
    4: "Neutral",
    5: "Sad"
}
app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="No input text provided")

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding='post')

    prediction = model.predict(padded)
    emotion_label = np.argmax(prediction)
    confidence = float(np.max(prediction))

    emotion_name = emotion_mapping.get(emotion_label, "Unknown")

    return {"emotion": emotion_name, "confidence": confidence}

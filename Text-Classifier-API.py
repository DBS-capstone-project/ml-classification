from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Dict
import tensorflow as tf
import numpy as np
import pickle
import json
import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

import requests
from supabase import create_client, Client

SUPABASE_URL = "Supabase url"
SUPABASE_KEY = "supabase jwt"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

model = tf.keras.models.load_model("emotion_classifier_model_rev2.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

EMOTIONS = ["Joy", "Sadness", "Fear", "Anger", "Love", "Neutral"]

emotion_mapping = {
    0: "Anger",
    1: "Fear",
    2: "Joy",
    3: "Love",
    4: "Neutral",
    5: "Sad"
}

class PredictionRequest(BaseModel):
    text: str
    user_choice: str
    question: str
    age_group: str

class EmotionInput(BaseModel):
    text: str
    predicted_emotion: str
    match: bool

class DailyMoodInput(BaseModel):
    mood: str
    reason: str
    text_reason: str

def check_prediction(user_choice: str, prediction_probs: np.ndarray, question: str, age_group: str):
    predicted_label = np.argmax(prediction_probs)
    predicted_emotion = emotion_mapping.get(predicted_label, "Unknown")
    
    is_matching = (user_choice == predicted_emotion)
    return predicted_emotion, predicted_emotion, is_matching
def log_emotion(entry: Dict):
    try:
        response = supabase.table("mood_log").insert(entry).execute()
        if hasattr(response, 'error') and response.error:
            print("Error logging emotion:", response.error)
    except Exception as e:
        print(f"Exception while logging emotion: {e}")

def load_logs():
    try:
        response = supabase.table("mood_log").select("*").execute()
        if hasattr(response, 'error') and response.error:
            print("Error loading logs:", response.error)
            return []
        return response.data
    except Exception as e:
        print(f"Exception while loading logs: {e}")
        return []

@app.post("/track-mood")
async def track_mood(entry: DailyMoodInput):
    input_text = entry.text_reason if entry.text_reason else entry.reason
    
    sequence = tokenizer.texts_to_sequences([input_text])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    prediction = model.predict(padded)[0]
    confidence = float(np.max(prediction))
    
    try:
        predicted_emotion, mapped_choice, is_match = check_prediction(
            entry.reason, prediction, "", ""
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    data = {
        "timestamp": datetime.now().isoformat(),
        "mood": entry.mood,
        "reason": entry.reason,
        "text_reason": entry.text_reason,
        "predicted_emotion": predicted_emotion,
        "match": is_match,
        "confidence": confidence
    }
    
    try:
        response = supabase.table("daily_mood").insert(data).execute()

        if hasattr(response, 'data'):
            return {
                "message": "Mood entry saved successfully",
                "predicted_emotion": predicted_emotion,
                "match": is_match,
                "confidence": confidence
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to log mood: Unknown error")
            
    except Exception as e:
        print(f"Error inserting data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to log mood: {str(e)}")

@app.get("/reflect")
async def reflect(days: int = 1):
    try:
        from_date = (datetime.now() - timedelta(days=days)).isoformat()
        response = supabase.table("daily_mood").select("*").gte("timestamp", from_date).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Tidak ada data mood yang ditemukan.")

        logs = response.data

        prompt = (
            "Kamu adalah teman refleksi yang baik, hangat, dan pengertian. "
            "Tugasmu adalah memberikan ringkasan singkat dari suasana hati pengguna selama beberapa hari terakhir "
            "berdasarkan data berikut ini. Jawaban maksimal 5 kalimat, jangan terlalu panjang. "
            "Tulis dengan gaya santai, pakai bahasa sehari-hari, tapi tetap empati. "
            "Berikan juga 1 saran yang ringan di akhir.\n\n"
            "Data:\n"
        )
        
        for log in logs:
            prompt += (
                f"- [{log.get('timestamp')}], mood: {log.get('mood')}, "
                f"alasan: {log.get('text_reason') or log.get('reason')}, "
                f"prediksi emosi: {log.get('predicted_emotion')}, "
                f"kecocokan: {'ya' if log.get('match') else 'tidak'}\n"
            )

        prompt += (
            "\nTolong berikan ringkasan dalam Bahasa Indonesia ya."
        )

        llama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False}
        )

        if llama_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Gagal mendapatkan respon dari LLaMA.")

        result = llama_response.json()["response"]

        return {"ringkasan_refleksi": result.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refleksi gagal: {str(e)}")

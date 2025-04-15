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
import requests
from typing import List
from tensorflow.keras.preprocessing.sequence import pad_sequences

from supabase import create_client, Client

SUPABASE_URL = "done"
SUPABASE_KEY = "done"
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

class ChatRequest(BaseModel):
    user_id: str
    message: str

class DailyMoodInput(BaseModel):
    user_id: str
    mood: str
    reason: str
    text_reason: str

class ReflectionRequest(BaseModel):
    user_id: str

class ReflectionResponse(BaseModel):
    user_id: str
    category: str
    questions: List[str]
    feedback: str

def check_prediction(user_choice: str, prediction_probs: np.ndarray, question: str, age_group: str):
    predicted_label = np.argmax(prediction_probs)
    predicted_emotion = emotion_mapping.get(predicted_label, "Unknown")
    is_matching = (user_choice == predicted_emotion)
    return predicted_emotion, is_matching

def detect_distress(text: str) -> bool:
    if not text:
        return False
    distress_keywords = [
        "aku udah capek", "aku gak kuat", "aku nyerah", "semua salahku",
        "gak ada yang peduli", "aku pengen hilang", "aku pengen pergi",
        "aku pengen mati", "gak tahan lagi", "cape banget", 
        "kenapa aku harus hidup", "udah cukup", "tolong aku",
        "aku benci hidupku", "gak ada gunanya", "aku sendirian",
        "aku takut", "aku kesepian", "aku gak sanggup lagi",
        "buat apa aku di sini", "aku gak punya harapan", "aku mau mati rasanya"
    ]
    text = text.lower()
    return any(phrase in text for phrase in distress_keywords)

@app.post("/chat")
async def chat(request: ChatRequest):
    history_resp = supabase.table("chat_history")\
        .select("user_message, ai_response")\
        .eq("user_id", request.user_id)\
        .order("timestamp", desc=True)\
        .limit(10)\
        .execute()
    history = history_resp.data or []

    mood_resp = supabase.table("mood_tracking")\
        .select("mood, created_at")\
        .eq("user_id", request.user_id)\
        .order("created_at", desc=True)\
        .limit(1)\
        .execute()
    mood_data = mood_resp.data

    if mood_data:
        last_mood = mood_data[0]
        mood_context = f"[Context: Mood terakhir user adalah '{last_mood['mood']}'.]\n"
    else:
        mood_context = "[Context: Belum ada data mood sebelumnya.]\n"

    convo_context = ""
    for pair in reversed(history):
        convo_context += f"User: {pair['user_message']}\nBot: {pair['ai_response']}\n"

    prompt = (
        f"{mood_context}"
        "Nama mu adalah Sparkle"
        "Apabila user menanyakan waktu, hindari menjawab waktu, jelaskan karena perbedaan waktu di dunia dan posisi kamu saat ini."
        "Kamu adalah teman ngobrol yang bisa membaca situasi hati. "
        "Jawablah dengan empati dan gaya santai, sesuaikan dengan mood user. "
        "Tanyakan juga apakah user ingin melanjutkan obrolan atau membicarakan hal baru.\n\n"
        f"{convo_context}User: {request.message}\nBot:"
    )

    llama_response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": True}
    )

    reply = ""
    for line in llama_response.iter_lines(decode_unicode=True):
        if line.strip():
            try:
                chunk = json.loads(line)
                reply += chunk.get("response", "")
            except json.JSONDecodeError:
                continue

    return {
        "reply": reply.strip(),
        "distress": detect_distress(request.message)
    }

@app.post("/mood-reflect")
async def mood_reflect(entry: DailyMoodInput, days: int = 1):
    input_text = entry.text_reason or entry.reason

    padded = pad_sequences(tokenizer.texts_to_sequences([input_text]), maxlen=100, padding='post')
    prediction = model.predict(padded)[0]
    confidence = float(np.max(prediction))

    try:
        predicted_emotion, is_match = check_prediction(entry.reason, prediction, "", "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        from_date = (datetime.now() - timedelta(days=days)).date().isoformat()

        response = (
            supabase.table("mood_tracking")
            .select("*")
            .eq("user_id", entry.user_id)
            .gte("date", from_date)
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail="Tidak ada data mood yang ditemukan.")

        logs = response.data

        summary_prompt = (
            "Kamu adalah teman refleksi yang santai tapi peduli. "
            "Tugasmu adalah membuat ringkasan super singkat dari data emosi pengguna. "
            "Gunakan bahasa sehari-hari yang to the point dan penuh empati. "
            "Jangan bertele-tele. Langsung aja ke intinya.\n\n"
            "Data:\n"
        )

        for log in logs:
            summary_prompt += (
                f"- [Tanggal: {log.get('date')}], mood: {log.get('mood')}, "
                f"alasan: {log.get('reason')}\n"
            )

        summary_prompt += "\nTolong berikan ringkasan dalam Bahasa Indonesia ya."

        reflection_prompt = (
            "Kamu adalah teman refleksi yang perhatian, sabar, dan suportif. "
            "Berikan refleksi singkat namun mendalam tentang data emosi hari ini. "
            "Sertakan saran ringan untuk meningkatkan kesejahteraan emosional pengguna. "
            "Gunakan bahasa yang santai dan empatik, maksimal 2-3 paragraf.\n\n"
            f"Emosi hari ini: {entry.mood}\n"
            f"Alasan: {input_text}\n\n"
            "Tolong buat refleksi dalam Bahasa Indonesia."
        )

        def call_llama(prompt: str):
            llama_resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2", "prompt": prompt, "stream": True}
            )
            if llama_resp.status_code != 200:
                raise HTTPException(status_code=500, detail="Gagal mendapatkan respon dari LLaMA.")
            result = ""
            for line in llama_resp.iter_lines(decode_unicode=True):
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        result += chunk.get("response", "")
                    except json.JSONDecodeError:
                        continue
            return result.strip()

        ringkasan = call_llama(summary_prompt)
        refleksi = call_llama(reflection_prompt)

        return {
            "message": "Mood processed & reflection generated",
            "predicted_emotion": predicted_emotion,
            "match": is_match,
            "confidence": confidence,
            "ringkasan_refleksi": ringkasan,
            "refleksi": refleksi
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refleksi gagal: {str(e)}")

@app.get("/weekly-summary/{user_id}")
async def weekly_summary(user_id: str):
    try:
        from_date = (datetime.now() - timedelta(days=7)).isoformat()
        logs_resp = (
            supabase.table("mood_tracking")
            .select("*")
            .eq("user_id", user_id)
            .gte("created_at", from_date)
            .order("created_at")
            .execute()
        )

        logs = logs_resp.data
        if not logs:
            raise HTTPException(status_code=404, detail="Belum ada data mood minggu ini.")

        prompt = (
            "Tolong buat ringkasan refleksi minggu ini dari data mood dan emosi user berikut. "
            "Gunakan bahasa Indonesia yang santai, to the point, dan penuh empati. "
            "Tambahkan satu saran ringan di akhir.\n\nData:\n"
        )

        for log in logs:
            prompt += (
                f"- [{log.get('created_at')}], mood: {log.get('mood')}, "
                f"alasan: {log.get('reason')}\n"
            )

        prompt += "\nLangsung aja ke intinya ya."

        llama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": True}
        )

        result = ""
        for line in llama_response.iter_lines(decode_unicode=True):
            if line.strip():
                try:
                    chunk = json.loads(line)
                    result += chunk.get("response", "")
                except json.JSONDecodeError:
                    continue

        return {
            "summary": result.strip()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membuat ringkasan mingguan: {str(e)}")

@app.post("/reflection-feedback", response_model=ReflectionResponse)
async def get_user_feedback(request: ReflectionRequest):
    try:
        user_id = request.user_id
        form_res = supabase.table("reflection_forms") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("timestamp", desc=True) \
            .limit(1) \
            .execute()
        
        if not form_res.data:
            raise HTTPException(status_code=404, detail="No reflection data found")

        form = form_res.data[0]
        category = form.get("category", "umum")
        questions = []
        for i in range(1, 11):
            if (q := form.get(f"question_{i}")):
                questions.append(q)

        if not questions:
            raise HTTPException(status_code=400, detail="No questions found in reflection data")

        prompt = (
            "Kamu adalah teman refleksi yang santai dan tidak kaku. "
            "Dari data pertanyaan berikut, buat ringkasan dan saran singkat. "
            "Tulis langsung jawabannya tanpa kata pembuka seperti 'Kesimpulan', 'Saran', atau 'Baiklah'. "
            "Gunakan bahasa sehari-hari, santai, dan hanya 2–3 kalimat saja.\n\n"
            "Data:\n"
        )

        for q in questions:
            prompt += f"\n- {q}"
        prompt += "\nLangsung aja ke intinya ya."

        llama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": True}
        )

        result = ""
        for line in llama_response.iter_lines(decode_unicode=True):
            if line.strip():
                try:
                    chunk = json.loads(line)
                    result += chunk.get("response", "")
                except json.JSONDecodeError:
                    continue

        return {
            "user_id": user_id,
            "category": category,
            "questions": questions,
            "feedback": result.strip()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

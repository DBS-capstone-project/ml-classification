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
from tensorflow.keras.preprocessing.sequence import pad_sequences

from supabase import create_client, Client

SUPABASE_URL = ""
SUPABASE_KEY = ""
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


def check_prediction(user_choice: str, prediction_probs: np.ndarray, question: str, age_group: str):
    predicted_label = np.argmax(prediction_probs)
    predicted_emotion = emotion_mapping.get(predicted_label, "Unknown")
    
    is_matching = (user_choice == predicted_emotion)
    return predicted_emotion, is_matching
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
    
@app.post("/chat")
async def chat(request: ChatRequest):
    history_resp = supabase.table("chat_history")\
        .select("message,response")\
        .eq("user_id", request.user_id)\
        .order("timestamp", desc=True)\
        .limit(10)\
        .execute()
    history = history_resp.data or []

    mood_resp = supabase.table("daily_mood")\
        .select("mood, predicted_emotion, timestamp")\
        .eq("user_id", request.user_id)\
        .order("timestamp", desc=True)\
        .limit(1)\
        .execute()
    mood_data = mood_resp.data

    if mood_data:
        last_mood = mood_data[0]
        mood_context = (
            f"[Context: Mood terakhir user adalah '{last_mood['mood']}' "
            f"dengan prediksi emosi '{last_mood['predicted_emotion']}'.]\n"
        )
    else:
        mood_context = "[Context: Belum ada data mood sebelumnya.]\n"

    convo_context = ""
    for pair in reversed(history):
        convo_context += f"User: {pair['message']}\nBot: {pair['response']}\n"

    is_distressed = detect_distress(request.message)

    # Buggy AF, Dont expect anything doe, but it is working.
    if is_distressed:
        instruction = (
            "Kamu adalah chatbot yang ngobrol santai dalam Bahasa Indonesia. Kamu menjawab sebagai teman baik, bukan seperti ensiklopedia."
            "User sepertinya sedang merasa sangat sedih, tertekan, atau tidak baik-baik saja. "
            "Responlah dengan sangat empatik, lembut, dan penuh perhatian. Jangan menyalahkan, dan jangan terlalu panjang. "
            "Ajak user untuk menarik napas sejenak, dan ingatkan bahwa kamu di sini untuk mereka. "
            "Ingatkan juga bahwa tidak apa-apa untuk bicara ke orang terdekat seperti keluarga, sahabat, atau profesional seperti konselor atau terapis. "
            "Kalau kamu ingin, kamu juga bisa menyebut layanan seperti MindsparkAi secara ringan dan sopan, tapi utamakan kenyamanan user. "
            "Buat mereka merasa tidak sendiri, dan tawarkan dukungan atau hal kecil yang menenangkan.\n"
        )
    else:
        instruction = (
            "Kamu adalah teman ngobrol yang bisa baca situasi hati. Jawablah dengan empati dan gaya santai, "
            "sesuai dengan mood user. Jangan terlalu kaku. Tanyakan juga apakah user ingin melanjutkan obrolan itu "
            "atau membicarakan hal baru.\n"
        )


    prompt = (
        f"{mood_context}"
        f"{instruction}\n"
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

    supabase.table("chat_history").insert({
        "user_id": request.user_id,
        "message": request.message,
        "response": reply
    }).execute()

    return {
        "reply": reply.strip(),
        "distress": is_distressed
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

    data = {
        "user_id": entry.user_id,
        "timestamp": datetime.now().isoformat(),
        "mood": entry.mood,
        "reason": entry.reason,
        "text_reason": entry.text_reason,
        "predicted_emotion": predicted_emotion,
        "match": is_match,
        "confidence": confidence,
        "distress_flag": detect_distress(input_text)
    }

    try:
        supabase.table("daily_mood").insert(data).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan mood: {str(e)}")

    try:
        from_date = (datetime.now() - timedelta(days=days)).isoformat()
        response = (
            supabase.table("daily_mood")
            .select("*")
            .eq("user_id", entry.user_id)
            .gte("timestamp", from_date)
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail="Tidak ada data mood yang ditemukan.")

        logs = response.data

        if not is_match:
            summary_prompt = (
                "Kamu adalah teman refleksi yang santai tapi peduli. "
                "Tugasmu adalah membuat ringkasan super singkat dari perasaan dan cerita pengguna. "
                "Gunakan bahasa sehari-hari yang santai, to the point, dan jangan bertele-tele. "
                "Tambahkan satu saran ringan di akhir yaa.\n\n"
                f"Emosi pengguna: {entry.mood}\n"
                f"Alasan: {input_text}\n\n"
                "Tolong berikan ringkasan dalam Bahasa Indonesia ya."
            )
        else:
            summary_prompt = (
                "Kamu adalah teman refleksi yang santai tapi peduli. "
                "Tugasmu adalah membuat ringkasan super singkat dari data emosi pengguna. "
                "Gunakan bahasa sehari-hari yang santai, to the point, dan jangan bertele-tele. "
                "Hindari kalimat pembuka seperti 'Berikut ringkasan...'. "
                "Langsung aja ke intinya ya!\n\n"
                "Data:\n"
            )
            for log in logs:
                summary_prompt += (
                    f"- [{log.get('timestamp')}], mood: {log.get('mood')}, "
                    f"alasan: {log.get('text_reason') or log.get('reason')}, "
                    f"prediksi emosi: {log.get('predicted_emotion')}, "
                    f"kecocokan: {'ya' if log.get('match') else 'tidak'}\n"
                )
            summary_prompt += "\nTolong berikan ringkasan dalam Bahasa Indonesia ya."

        reflection_prompt = (
            "Kamu adalah teman refleksi yang perhatian, sabar, dan suportif. "
            "Tugasmu adalah memberikan refleksi yang ringan tapi membantu pengguna memahami perasaannya. "
            "Berikan saran singkat, validasi emosinya, dan beri dukungan ringan. "
            "Gunakan gaya bahasa santai dan empatik, hindari bahasa formal ya.\n\n"
            f"Emosi pengguna hari ini: {entry.mood}\n"
            f"Alasannya: {input_text}\n\n"
            "Tolong buat refleksi dalam Bahasa Indonesia ya, jangan terlalu panjang, dan cukup 2-3 paragraf ringan."
        )

        def call_llama(prompt: str):
            llama_response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2", "prompt": prompt, "stream": True}
            )

            if llama_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Gagal mendapatkan respon dari LLaMA.")

            result = ""
            for line in llama_response.iter_lines(decode_unicode=True):
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
            "message": "Mood saved & reflection generated",
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
            supabase.table("daily_mood")
            .select("*")
            .eq("user_id", user_id)
            .gte("timestamp", from_date)
            .order("timestamp")
            .execute()
        )

        logs = logs_resp.data
        if not logs:
            raise HTTPException(status_code=404, detail="Belum ada data mood minggu ini.")

        # Build LLaMA prompt
        prompt = (
            "Tolong buat ringkasan refleksi minggu ini dari data mood dan emosi user berikut. "
            "Gunakan bahasa Indonesia yang santai, to the point, dan penuh empati. "
            "Tambahkan satu saran ringan di akhir.\n\nData:\n"
        )

        for log in logs:
            prompt += (
                f"- [{log.get('timestamp')}], mood: {log.get('mood')}, "
                f"alasan: {log.get('text_reason') or log.get('reason')}, "
                f"prediksi emosi: {log.get('predicted_emotion')}, "
                f"kecocokan: {'ya' if log.get('match') else 'tidak'}\n"
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


@app.get("/user-reflection-feedback/{user_id}")
async def get_user_feedback(user_id: str):
    try:

        form_res = supabase.table("reflection_forms") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("timestamp", desc=True) \
            .limit(1) \
            .execute()
        
        if not form_res.data:
            raise HTTPException(status_code=404, detail="No reflection data found")

        form = form_res.data[0]

        prompt = (
            "Kamu adalah teman refleksi yang santai dan tidak kaku."
            "Dari data pertanyaan berikut, buat ringkasan dan saran singkat."
            "Tulis langsung jawabannya tanpa kata pembuka seperti 'Kesimpulan', 'Saran', atau 'Baiklah'."
            "Gunakan bahasa sehari-hari, santai, dan hanya 2–3 kalimat saja.\n\n"
            "Data:\n"
        )

        for i in range(1, 11):
            if q := form.get(f"question_{i}"):
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
            "feedback": result.strip()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

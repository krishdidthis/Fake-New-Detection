import yt_dlp
import whisper
import os

def download_audio(youtube_url, output_path="temp_audio"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path + ".mp3"

# ---- Test ----
youtube_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # koi chhota video link daal do yahan

print("Downloading audio...")
audio_file = download_audio(youtube_url)

print("Transcribing with Whisper...")
model = whisper.load_model("base")
result = model.transcribe(audio_file)

print("\n--- Transcribed Text ---")
print(result["text"])

import pickle
import re
from nltk.corpus import stopwords

# Saved model load karo
model_fnd = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

# Whisper se mila transcript use karo
transcribed_text = result["text"]

cleaned = clean_text(transcribed_text)
vec = vectorizer.transform([cleaned])
prediction = model_fnd.predict(vec)[0]
prob = model_fnd.predict_proba(vec)[0]

label = "REAL" if prediction == 1 else "FAKE"
confidence = round(max(prob) * 100, 2)

print(f"\n--- Final Prediction ---")
print(f"Video Transcript: {transcribed_text}")
print(f"Prediction: {label} (Confidence: {confidence}%)")
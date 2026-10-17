from flask import Flask, render_template, request, jsonify
import pickle
import re
import os
from nltk.corpus import stopwords
import yt_dlp
import whisper


app = Flask(__name__)


# =========================
# LOAD SAVED MODEL
# =========================

model = pickle.load(
    open("model/fake_news_model.pkl", "rb")
)

vectorizer = pickle.load(
    open("model/vectorizer.pkl", "rb")
)

stop_words = set(
    stopwords.words("english")
)


# =========================
# LOAD WHISPER MODEL
# =========================

whisper_model = whisper.load_model("base")


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(
        r'https?://\S+|www\.\S+',
        '',
        text
    )

    text = re.sub(
        r'<.*?>+',
        '',
        text
    )

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    text = ' '.join(
        [
            word
            for word in text.split()
            if word not in stop_words
        ]
    )

    return text


# =========================
# DOWNLOAD YOUTUBE AUDIO
# =========================

def download_audio(
    youtube_url,
    output_path="temp_audio"
):

    ydl_opts = {

        "format": "bestaudio/best",

        "outtmpl":
            output_path + ".%(ext)s",

        "postprocessors": [

            {
                "key":
                    "FFmpegExtractAudio",

                "preferredcodec":
                    "mp3",

                "preferredquality":
                    "192"
            }
        ]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        ydl.download([youtube_url])

    return output_path + ".mp3"


# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================
# TEXT PREDICTION
# =========================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    data = request.get_json()

    news_text = data.get(
        "news",
        ""
    )

    if not news_text.strip():

        return jsonify({
            "error":
                "Please enter some news text."
        }), 400


    # Clean text

    cleaned = clean_text(
        news_text
    )


    # TF-IDF

    vec = vectorizer.transform(
        [cleaned]
    )


    # Prediction

    prediction = model.predict(
        vec
    )[0]


    # Probability

    prob = model.predict_proba(
        vec
    )[0]


    # Label

    label = (
        "REAL"
        if prediction == 1
        else "FAKE"
    )


    # Confidence

    confidence = round(
        max(prob) * 100,
        2
    )


    return jsonify({

        "label": label,

        "confidence":
            confidence

    })


# =========================
# VIDEO PREDICTION
# =========================

@app.route(
    "/predict_video",
    methods=["POST"]
)
def predict_video():

    data = request.get_json()

    youtube_url = data.get(
        "video_url",
        ""
    ).strip()


    # Check URL

    if not youtube_url:

        return jsonify({
            "error":
                "Please provide a YouTube URL."
        }), 400


    audio_file = None


    try:

        # -------------------------
        # STEP 1
        # Download video audio
        # -------------------------

        print(
            "\nDownloading video audio..."
        )

        audio_file = download_audio(
            youtube_url
        )


        # -------------------------
        # STEP 2
        # Whisper transcription
        # -------------------------

        print(
            "Transcribing video with Whisper..."
        )

        result = whisper_model.transcribe(
            audio_file
        )

        transcript = result["text"]


        # Check transcript

        if not transcript.strip():

            return jsonify({
                "error":
                    "Could not detect speech in the video."
            }), 400


        print(
            "\nTranscript:"
        )

        print(
            transcript
        )


        # -------------------------
        # STEP 3
        # Clean transcript
        # -------------------------

        cleaned = clean_text(
            transcript
        )


        # -------------------------
        # STEP 4
        # TF-IDF
        # -------------------------

        vec = vectorizer.transform(
            [cleaned]
        )


        # -------------------------
        # STEP 5
        # Fake / Real prediction
        # -------------------------

        prediction = model.predict(
            vec
        )[0]


        prob = model.predict_proba(
            vec
        )[0]


        # -------------------------
        # STEP 6
        # Label
        # -------------------------

        label = (
            "REAL"
            if prediction == 1
            else "FAKE"
        )


        # -------------------------
        # STEP 7
        # Confidence
        # -------------------------

        confidence = round(
            max(prob) * 100,
            2
        )


        # -------------------------
        # STEP 8
        # Delete temporary audio
        # -------------------------

        if (
            audio_file
            and os.path.exists(
                audio_file
            )
        ):

            os.remove(
                audio_file
            )


        # -------------------------
        # STEP 9
        # Send result to website
        # -------------------------

        return jsonify({

            "label":
                label,

            "confidence":
                confidence,

            "transcript":
                transcript

        })


    except Exception as e:


        print(
            "\nVIDEO ERROR:"
        )

        print(
            str(e)
        )


        # Remove temporary file

        if (
            audio_file
            and os.path.exists(
                audio_file
            )
        ):

            os.remove(
                audio_file
            )


        return jsonify({

            "error":
                str(e)

        }), 500


# =========================
# START FLASK SERVER
# =========================

if __name__ == "__main__":

    app.run(
        debug=True
    )
import os
import uuid
import traceback
import re
from pathlib import Path

import requests
from flask import Flask, render_template, request, jsonify
from faster_whisper import WhisperModel

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
WHISPER_MODEL_SIZE = "base"

USE_CUDA = False

if USE_CUDA:
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
else:
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")



def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def transcribe_audio(audio_path: str) -> str:
    segments, info = whisper_model.transcribe(
        audio_path,
        language="en",
        vad_filter=True
    )

    texts = []
    for seg in segments:
        if seg.text:
            texts.append(seg.text.strip())

    return " ".join(texts).strip()


def ask_llama3(target_text: str, recognized_text: str) -> str:
    normalized_target = normalize_text(target_text)
    normalized_recognized = normalize_text(recognized_text)

    prompt = f"""
You are an English pronunciation coach.

Original target sentence:
"{target_text}"

Original recognized sentence:
"{recognized_text}"

Normalized target sentence:
"{normalized_target}"

Normalized recognized sentence:
"{normalized_recognized}"

Rules:
1. Ignore capitalization differences.
2. Ignore punctuation differences.
3. Focus only on possible pronunciation-related differences.
4. If the normalized recognized sentence is very close to the normalized target sentence, say the pronunciation is generally good.
5. Use simple English.
6. Give at most 2 short practice tips.
7. Keep the response concise and friendly.

Return plain text only.
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        target_text = request.form.get("target_text", "").strip()
        audio_file = request.files.get("audio")

        print("target_text =", target_text)
        print("audio_file =", audio_file.filename if audio_file else None)

        if not target_text:
            return jsonify({"error": "target_text is required"}), 400

        if audio_file is None:
            return jsonify({"error": "audio file is required"}), 400

        suffix = Path(audio_file.filename).suffix.lower() or ".webm"
        save_name = f"{uuid.uuid4().hex}{suffix}"
        save_path = UPLOAD_DIR / save_name
        audio_file.save(save_path)

        print("saved audio to:", save_path)

        recognized_text = transcribe_audio(str(save_path))
        print("recognized_text =", recognized_text)

        if not recognized_text:
            feedback = "I could not recognize clear speech from the audio. Please try again in a quieter environment."
        else:
            feedback = ask_llama3(target_text, recognized_text)

        print("feedback =", feedback)

        return jsonify({
            "target_text": target_text,
            "recognized_text": recognized_text,
            "feedback": feedback
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
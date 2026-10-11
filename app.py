import os
import subprocess
import tempfile
import requests
import birdnet
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# URL and model name for the local Ollama instance
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

# Load the BirdNET model ONCE at startup
print("Loading BirdNET model at startup...")
model = birdnet.load("acoustic", "2.4", "tf", library="litert")
print("BirdNET model loaded successfully.")


def get_field_note(common_name: str, confidence: float) -> str:
    """
    Sends species name and confidence to local Ollama API to generate a
    2-sentence beginner-friendly field note. If confidence is below 0.4,
    indicates that the match is uncertain.
    """
    if confidence < 0.4:
        prompt = (
            f"Write a concise, 2-sentence beginner-friendly birdwatching field note "
            f"for the bird species '{common_name}'. Note that this audio identification "
            f"is uncertain (low confidence: {confidence:.2f}), and mention key visual traits "
            f"or call characteristics birders should look for to confirm the identification."
        )
    else:
        prompt = (
            f"Write a concise, 2-sentence beginner-friendly birdwatching field note "
            f"for the bird species '{common_name}'. Focus on distinguishing visual traits "
            f"or common behaviors."
        )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,  # Receive complete response in one JSON payload
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            f"Could not connect to Ollama at {OLLAMA_API_URL}. "
            "Please ensure Ollama is running locally."
        )
    except requests.exceptions.RequestException as e:
        return f"Error generating field note: {str(e)}"


@app.route("/")
def index():
    """Renders the main upload page."""
    return render_template("index.html")


@app.route("/identify", methods=["POST"])
def identify():
    """
    Accepts an uploaded audio file, converts webm/ogg to wav via ffmpeg if needed,
    predicts bird species using BirdNET, and generates a field note via Ollama.
    """
    # Verify that an audio file was provided in the request
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided in request"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "No audio file selected"}), 400

    # Save the uploaded audio to a temporary file
    file_ext = os.path.splitext(audio_file.filename)[1] or ".wav"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    temp_path = temp_file.name
    temp_file.close()

    wav_path = None
    try:
        audio_file.save(temp_path)

        predict_path = temp_path

        # If audio is webm/ogg (common from browser MediaRecorder) or not standard wav,
        # convert it to .wav using ffmpeg so BirdNET can read it properly.
        if file_ext.lower() in [".webm", ".ogg", ".opus", ".m4a"]:
            wav_path = temp_path + ".wav"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", temp_path, "-ar", "48000", "-ac", "1", wav_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                predict_path = wav_path
            except (subprocess.SubprocessError, FileNotFoundError):
                predict_path = temp_path

        # Run BirdNET prediction on the audio file
        try:
            predictions = model.predict(predict_path)
        except Exception as pred_err:
            # If BirdNET failed on the original audio format, attempt ffmpeg conversion as fallback
            if predict_path == temp_path and not wav_path:
                wav_path = temp_path + ".wav"
                subprocess.run(
                    ["ffmpeg", "-y", "-i", temp_path, "-ar", "48000", "-ac", "1", wav_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                predictions = model.predict(wav_path)
            else:
                raise pred_err

        df = predictions.to_dataframe()

        # If no birds detected, skip Ollama and return immediately
        if df is None or df.empty:
            return jsonify({
                "species": "No bird detected",
                "scientific_name": None,
                "confidence": None,
                "field_note": None
            })

        # Pick the row with the highest confidence
        best_row = df.loc[df["confidence"].idxmax()]
        species_raw = str(best_row["species_name"])
        confidence = float(best_row["confidence"])

        # Split species_name on the first underscore into scientific and common name
        if "_" in species_raw:
            scientific_name, common_name = species_raw.split("_", 1)
        else:
            scientific_name, common_name = "", species_raw

        # Generate field note via Ollama (uncertain if confidence < 0.4)
        field_note = get_field_note(common_name, confidence)

        return jsonify({
            "species": common_name,
            "scientific_name": scientific_name,
            "confidence": round(confidence, 4),
            "field_note": field_note
        })

    finally:
        # Clean up temporary files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True, use_reloader=False)

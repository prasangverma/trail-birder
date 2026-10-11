import os
import subprocess
import tempfile
import shutil
import numpy as np
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


def print_audio_stats(audio_path: str) -> None:
    """
    Inspects audio file using ffmpeg volumedetect filter and prints
    duration, peak (max) volume, and mean volume in the terminal.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True,
            text=True
        )
        stderr = result.stderr
        duration = "Unknown"
        max_volume = "Unknown"
        mean_volume = "Unknown"

        for line in stderr.splitlines():
            if "Duration:" in line:
                duration = line.split("Duration:")[1].split(",")[0].strip()
            if "max_volume:" in line:
                max_volume = line.split("max_volume:")[1].strip()
            if "mean_volume:" in line:
                mean_volume = line.split("mean_volume:")[1].strip()

        print("\n" + "=" * 45)
        print(" [AUDIO STATS - RECORDING INSPECTION]")
        print(f"  File: {audio_path}")
        print(f"  Duration: {duration}")
        print(f"  Peak Volume (max_volume): {max_volume}")
        print(f"  Mean Volume (mean_volume): {mean_volume}")
        print("=" * 45 + "\n", flush=True)
    except Exception as e:
        print(f"[AUDIO STATS] Could not inspect volume stats: {e}", flush=True)


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
    Accepts an uploaded audio file, converts it with ffmpeg (boosting quiet audio),
    saves to debug/last_recording.wav, predicts bird species with BirdNET (confidence threshold 0.1),
    and falls back to top guess if detection is below threshold.
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

        # Boost quiet audio and convert to 48kHz mono WAV:
        # ffmpeg -y -i input -af "loudnorm,highpass=f=300" -ar 48000 -ac 1 output.wav
        wav_path = temp_path + ".wav"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", temp_path,
                    "-af", "loudnorm,highpass=f=300",
                    "-ar", "48000", "-ac", "1",
                    wav_path
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            predict_path = wav_path
        except (subprocess.SubprocessError, FileNotFoundError) as conv_err:
            print(f"[Warning] ffmpeg boost conversion failed: {conv_err}. Using original file.")
            predict_path = temp_path

        # Save each converted recording to debug/last_recording.wav
        os.makedirs("debug", exist_ok=True)
        debug_wav_path = os.path.join("debug", "last_recording.wav")
        if os.path.exists(predict_path):
            shutil.copy2(predict_path, debug_wav_path)
            # Print duration and volume stats to terminal
            print_audio_stats(debug_wav_path)

        # Predict using BirdNET with default_confidence_threshold lowered to 0.1
        predictions = model.predict(predict_path, default_confidence_threshold=0.1)
        df = predictions.to_dataframe()

        note = None
        # If the detection is still empty, show the top guess anyway with a low confidence note
        if df is None or df.empty:
            probs = predictions.species_probs
            ids = predictions.species_ids
            if probs.size > 0 and ids.size > 0:
                max_flat_idx = int(np.argmax(probs))
                max_tuple_idx = np.unravel_index(max_flat_idx, probs.shape)
                top_id = ids[max_tuple_idx]
                confidence = float(probs[max_tuple_idx])
                species_raw = str(predictions.species_list[top_id])
                note = "Low confidence match (top guess below 0.1 threshold)"
            else:
                return jsonify({
                    "species": "No bird detected",
                    "scientific_name": None,
                    "confidence": None,
                    "field_note": None
                })
        else:
            # Pick the row with the highest confidence
            best_row = df.loc[df["confidence"].idxmax()]
            species_raw = str(best_row["species_name"])
            confidence = float(best_row["confidence"])

        # Split species_name on the first underscore into scientific and common name
        if "_" in species_raw:
            scientific_name, common_name = species_raw.split("_", 1)
        else:
            scientific_name, common_name = "", species_raw

        # Generate field note via Ollama (marks uncertain if confidence < 0.4)
        field_note = get_field_note(common_name, confidence)

        return jsonify({
            "species": common_name,
            "scientific_name": scientific_name,
            "confidence": round(confidence, 4),
            "field_note": field_note,
            "note": note
        })

    finally:
        # Clean up temporary processing files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True, use_reloader=False)

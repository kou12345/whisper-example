from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile
import os
from typing import Dict, List
import torch
from pyannote.audio import Pipeline
import librosa
import numpy as np

app = FastAPI(title="Whisper Transcription API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

whisper_model_1 = None
whisper_model_2 = None
diarization_pipeline = None
current_model_index = 0
model_lock = {"1": False, "2": False}


@app.on_event("startup")
async def startup_event():
    global whisper_model_1, whisper_model_2, diarization_pipeline
    print("Loading Faster-Whisper models (2 instances)...")
    whisper_model_1 = WhisperModel("medium", device="cpu", compute_type="int8")
    whisper_model_2 = WhisperModel("medium", device="cpu", compute_type="int8")
    print("Both Whisper models loaded successfully")

    try:
        print("Loading speaker diarization pipeline...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=None
        )
        if torch.cuda.is_available():
            diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
    except Exception as e:
        print(f"Warning: Could not load diarization pipeline: {e}")
        diarization_pipeline = None


@app.get("/")
async def root():
    return {"message": "Whisper Transcription API is running"}


def get_available_model():
    global current_model_index, model_lock, whisper_model_1, whisper_model_2
    
    # Try to get model 1 if available
    if not model_lock["1"]:
        model_lock["1"] = True
        return whisper_model_1, "1"
    
    # Try to get model 2 if available
    if not model_lock["2"]:
        model_lock["2"] = True
        return whisper_model_2, "2"
    
    # If both are busy, wait for model 1 (round-robin)
    model_lock["1"] = True
    return whisper_model_1, "1"

def release_model(model_id):
    global model_lock
    model_lock[model_id] = False

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    model_id = None
    try:
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be an audio file")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Get available model
            whisper_model, model_id = get_available_model()
            print(f"Using Whisper model {model_id}")
            
            segments, info = whisper_model.transcribe(temp_path, language="ja")

            # Convert faster-whisper segments to dictionary format
            segment_list = []
            full_text = ""
            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                segment_list.append(segment_dict)
                full_text += segment.text

            result = {
                "text": full_text,
                "segments": segment_list,
                "language": info.language,
            }

            # Filter out common hallucinated phrases
            unwanted_phrases = [
                "ご視聴ありがとうございました",
                "ご視聴ありがとうございます",
                "チャンネル登録お願いします",
                "お疲れ様でした",
                "ありがとうございました",
                "よろしくお願いします",
                "以上です",
                "Thank you for watching",
                "Please subscribe",
                "Thanks for watching",
            ]

            # Filter main text
            filtered_text = result["text"].strip()
            for phrase in unwanted_phrases:
                if phrase in filtered_text and len(filtered_text) <= len(phrase) + 5:
                    filtered_text = ""
                    break

            result["text"] = filtered_text

            speakers = []
            if diarization_pipeline:
                try:
                    diarization = diarization_pipeline(temp_path)
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        speakers.append(
                            {"start": turn.start, "end": turn.end, "speaker": speaker}
                        )
                except Exception as e:
                    print(f"Diarization failed: {e}")

            segments_with_speakers = []
            for segment in result["segments"]:
                # Filter segment text
                segment_text = segment["text"].strip()
                for phrase in unwanted_phrases:
                    if phrase in segment_text and len(segment_text) <= len(phrase) + 5:
                        segment_text = ""
                        break

                # Skip empty segments
                if not segment_text:
                    continue

                speaker = "Unknown"
                segment_start = segment["start"]
                segment_end = segment["end"]

                for spk in speakers:
                    if (
                        spk["start"] <= segment_start <= spk["end"]
                        or spk["start"] <= segment_end <= spk["end"]
                    ):
                        speaker = spk["speaker"]
                        break

                segments_with_speakers.append(
                    {
                        "start": segment_start,
                        "end": segment_end,
                        "text": segment_text,
                        "speaker": speaker,
                    }
                )

            return {
                "text": result["text"],
                "segments": segments_with_speakers,
                "language": result["language"],
            }

        finally:
            # Release the model
            if model_id:
                release_model(model_id)
            os.unlink(temp_path)

    except Exception as e:
        # Release the model in case of error
        if model_id:
            release_model(model_id)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "whisper_models_loaded": whisper_model_1 is not None and whisper_model_2 is not None,
        "diarization_available": diarization_pipeline is not None,
        "model_1_busy": model_lock["1"],
        "model_2_busy": model_lock["2"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

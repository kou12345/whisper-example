import mlx_whisper
import numpy as np
import logging
import asyncio
from typing import Dict, List, Optional
from config import *
from diarization import SpeakerDiarization

logger = logging.getLogger(__name__)

class MLXTranscriber:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.language = LANGUAGE
        self.is_loaded = False
        self.diarization = SpeakerDiarization() if ENABLE_DIARIZATION else None
        
    async def load_model(self):
        """Initialize MLX Whisper and Speaker Diarization (models are loaded on-demand)"""
        try:
            logger.info(f"MLX Whisper ready with model: {self.model_name}")
            self.is_loaded = True
            
            # Load diarization model if enabled
            if self.diarization:
                await self.diarization.load_model()
                logger.info("Speaker diarization initialized")
            
            logger.info(f"MLX Whisper initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MLX Whisper: {e}")
            raise

    async def transcribe(self, audio_data: np.ndarray) -> Dict:
        """Transcribe audio using MLX Whisper"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Ensure audio is float32 and normalize
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio to [-1, 1] range
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            logger.debug(f"Transcribing audio chunk: {len(audio_data)} samples")
            
            # Run transcription in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                audio_data
            )
            
            # Perform speaker diarization if enabled
            if self.diarization and result.get("segments"):
                logger.debug("Running speaker diarization")
                diarization_result = self.diarization.diarize_audio(audio_data)
                
                # Assign speakers to transcription segments
                result["segments"] = self.diarization.assign_speakers_to_transcription(
                    result["segments"], diarization_result
                )
                
                # Add speaker summary
                result["speaker_summary"] = self.diarization.get_speaker_summary(result["segments"])
                result["diarization"] = diarization_result
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                "text": "",
                "segments": [],
                "language": self.language
            }

    def _transcribe_sync(self, audio_data: np.ndarray) -> Dict:
        """Synchronous transcription wrapper"""
        try:
            result = mlx_whisper.transcribe(
                audio_data,
                path_or_hf_repo=self.model_name,
                language=self.language,
                word_timestamps=True
            )
            
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
                "Thanks for watching"
            ]
            
            # Filter main text
            filtered_text = result["text"].strip()
            for phrase in unwanted_phrases:
                if phrase in filtered_text and len(filtered_text) <= len(phrase) + 5:
                    filtered_text = ""
                    break
            
            # Filter segments
            filtered_segments = []
            for segment in result.get("segments", []):
                segment_text = segment["text"].strip()
                
                # Skip unwanted phrases
                skip_segment = False
                for phrase in unwanted_phrases:
                    if phrase in segment_text and len(segment_text) <= len(phrase) + 5:
                        skip_segment = True
                        break
                
                if not skip_segment and segment_text:
                    filtered_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment_text,
                        "speaker": "Unknown"  # Will be assigned by diarization
                    })
            
            return {
                "text": filtered_text,
                "segments": filtered_segments,
                "language": result.get("language", self.language)
            }
            
        except Exception as e:
            logger.error(f"MLX transcription error: {e}")
            return {
                "text": "",
                "segments": [],
                "language": self.language
            }

    def unload_model(self):
        """Cleanup resources"""
        self.is_loaded = False
        if self.diarization:
            self.diarization.unload_model()
        logger.info("MLX Whisper and diarization cleaned up")
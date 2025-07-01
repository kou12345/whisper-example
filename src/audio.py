import sounddevice as sd
import webrtcvad
import numpy as np
import asyncio
import logging
from typing import Generator, Optional
from collections import deque
from config import *

logger = logging.getLogger(__name__)

class AudioCapture:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.chunk_size = get_chunk_size()
        self.overlap_size = get_overlap_size()
        self.vad_frame_size = get_vad_frame_size()
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.chunk_size + self.overlap_size)
        self.is_recording = False
        
        logger.info(f"AudioCapture initialized: {self.sample_rate}Hz, chunk={self.chunk_size}, overlap={self.overlap_size}")

    def _is_voiced(self, audio_data: np.ndarray) -> bool:
        """Check if audio contains voice using WebRTC VAD"""
        try:
            # Convert to 16-bit PCM
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Process in VAD frame chunks
            voiced_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_int16) - self.vad_frame_size, self.vad_frame_size):
                frame = audio_int16[i:i + self.vad_frame_size]
                if len(frame) == self.vad_frame_size:
                    try:
                        if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                            voiced_frames += 1
                        total_frames += 1
                    except Exception as e:
                        logger.warning(f"VAD frame processing error: {e}")
                        continue
            
            # Consider voiced if >10% of frames contain speech (lowered threshold)
            if total_frames > 0:
                voice_ratio = voiced_frames / total_frames
                logger.debug(f"Voice ratio: {voice_ratio:.2f} ({voiced_frames}/{total_frames})")
                return voice_ratio > 0.1  # Lowered from 0.3 to 0.1
            return False
            
        except Exception as e:
            logger.error(f"VAD processing error: {e}")
            return True  # Assume voiced on error

    async def start_recording(self) -> Generator[np.ndarray, None, None]:
        """Start non-blocking audio recording with VAD filtering"""
        self.is_recording = True
        logger.info("Starting audio recording")
        
        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=BLOCKSIZE,
                callback=self._audio_callback
            ) as stream:
                
                logger.info(f"Audio stream started: {stream}")
                
                while self.is_recording:
                    await asyncio.sleep(CHUNK_MS / 1000)  # Wait for chunk duration
                    
                    logger.debug(f"Buffer size: {len(self.audio_buffer)}, required: {self.chunk_size}")
                    
                    if len(self.audio_buffer) >= self.chunk_size:
                        # Extract current chunk
                        chunk = np.array(list(self.audio_buffer)[:self.chunk_size])
                        
                        # Check if chunk contains voice
                        audio_level = np.max(np.abs(chunk))
                        logger.info(f"Audio level: {audio_level:.4f}")
                        
                        if self._is_voiced(chunk) or audio_level > 0.001:  # Also check basic audio level
                            logger.info(f"Voice/Audio detected, chunk size: {len(chunk)}")
                            yield chunk
                        else:
                            logger.info(f"No voice detected, skipping chunk (max level: {audio_level:.4f})")
                        
                        # Remove processed samples, keep overlap
                        for _ in range(self.chunk_size - self.overlap_size):
                            if self.audio_buffer:
                                self.audio_buffer.popleft()
                    else:
                        logger.debug(f"Waiting for more audio data...")
                                
        except Exception as e:
            logger.error(f"Recording error: {e}")
            raise
        finally:
            self.is_recording = False
            logger.info("Audio recording stopped")

    def _audio_callback(self, indata, frames, time, status):
        """Audio input callback"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        logger.debug(f"Audio callback received {frames} frames")
        
        # Convert CFFI buffer to numpy array for RawInputStream
        audio_data = np.frombuffer(indata, dtype=np.float32)
        logger.debug(f"Adding {len(audio_data)} samples to buffer")
        self.audio_buffer.extend(audio_data)

    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        logger.info("Stopping audio recording")

    def get_device_info(self):
        """Get audio device information"""
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            logger.info(f"Default input device: {devices[default_input]['name']}")
            return devices
        except Exception as e:
            logger.error(f"Error querying audio devices: {e}")
            return None
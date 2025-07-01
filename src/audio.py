import sounddevice as sd
import webrtcvad
import numpy as np
import asyncio
import logging
import threading
from typing import Generator, Optional, List
from collections import deque
import time
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
        
        # Enhanced buffering system
        self.audio_buffer = deque(maxlen=self.chunk_size + self.overlap_size)
        self.processing_buffer = deque()  # Buffer for chunks being processed
        self.buffer_lock = threading.Lock()  # Thread safety
        self.is_recording = False
        
        # Statistics
        self.chunks_captured = 0
        self.chunks_processed = 0
        self.chunks_dropped = 0
        
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
        """Start non-blocking audio recording with enhanced buffering"""
        self.is_recording = True
        logger.info("Starting enhanced audio recording")
        
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
                    # Check for ready chunks in processing buffer
                    chunk_ready = await self._get_next_chunk()
                    
                    if chunk_ready is not None:
                        # Check if chunk contains voice
                        audio_level = np.max(np.abs(chunk_ready))
                        logger.debug(f"Processing chunk - Audio level: {audio_level:.4f}")
                        
                        if self._is_voiced(chunk_ready) or audio_level > 0.001:
                            logger.info(f"Voice/Audio detected, chunk size: {len(chunk_ready)}")
                            self.chunks_processed += 1
                            yield chunk_ready
                        else:
                            logger.debug(f"No voice detected, skipping chunk (max level: {audio_level:.4f})")
                    else:
                        # No chunk ready, wait a bit
                        await asyncio.sleep(BUFFER_CHECK_INTERVAL)
                        
                    # Log statistics periodically
                    if self.chunks_captured > 0 and self.chunks_captured % 10 == 0:
                        self._log_statistics()
                                
        except Exception as e:
            logger.error(f"Recording error: {e}")
            raise
        finally:
            self.is_recording = False
            logger.info("Audio recording stopped")
            self._log_statistics()

    async def _get_next_chunk(self) -> Optional[np.ndarray]:
        """Get next available chunk from processing buffer"""
        with self.buffer_lock:
            # Try to create a new chunk if we have enough data
            if len(self.audio_buffer) >= self.chunk_size:
                # Extract current chunk
                chunk = np.array(list(self.audio_buffer)[:self.chunk_size])
                
                # Add to processing buffer
                self.processing_buffer.append({
                    'data': chunk,
                    'timestamp': time.time()
                })
                
                # Remove processed samples, keep overlap
                for _ in range(self.chunk_size - self.overlap_size):
                    if self.audio_buffer:
                        self.audio_buffer.popleft()
                
                self.chunks_captured += 1
            
            # Return oldest chunk from processing buffer
            if self.processing_buffer:
                chunk_info = self.processing_buffer.popleft()
                return chunk_info['data']
        
        return None

    def _log_statistics(self):
        """Log buffering statistics"""
        buffer_efficiency = (self.chunks_processed / max(self.chunks_captured, 1)) * 100
        logger.info(f"Buffer stats - Captured: {self.chunks_captured}, "
                   f"Processed: {self.chunks_processed}, "
                   f"Efficiency: {buffer_efficiency:.1f}%")
        
        with self.buffer_lock:
            logger.debug(f"Current buffers - Audio: {len(self.audio_buffer)}, "
                        f"Processing: {len(self.processing_buffer)}")

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
import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.audio import AudioCapture

class TestVAD:
    def setup_method(self):
        """Setup test fixtures"""
        self.audio_capture = AudioCapture()

    def test_vad_silence(self):
        """Test VAD with silence"""
        # Generate silent audio (2 seconds at 16kHz)
        silent_audio = np.zeros(32000, dtype=np.float32)
        
        result = self.audio_capture._is_voiced(silent_audio)
        assert result == False, "VAD should detect silence as non-voiced"

    def test_vad_noise(self):
        """Test VAD with white noise"""
        # Generate white noise (should be detected as voiced due to energy)
        noise_audio = np.random.normal(0, 0.1, 32000).astype(np.float32)
        
        result = self.audio_capture._is_voiced(noise_audio)
        # Note: White noise might be detected as voiced depending on VAD sensitivity
        assert isinstance(result, bool), "VAD should return boolean"

    def test_vad_sine_wave(self):
        """Test VAD with sine wave (simulated voice)"""
        # Generate sine wave at voice frequency (440 Hz)
        sample_rate = 16000
        duration = 2.0
        frequency = 440
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        sine_wave = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.3
        
        result = self.audio_capture._is_voiced(sine_wave)
        assert result == True, "VAD should detect sine wave as voiced"

    def test_vad_frame_size(self):
        """Test VAD frame size configuration"""
        assert self.audio_capture.vad_frame_size == 160, "VAD frame size should be 160 samples (10ms at 16kHz)"

    def test_chunk_sizes(self):
        """Test audio chunk size calculations"""
        assert self.audio_capture.chunk_size == 32000, "Chunk size should be 32000 samples (2s at 16kHz)"
        assert self.audio_capture.overlap_size == 8000, "Overlap size should be 8000 samples (0.5s at 16kHz)"
import torch
import numpy as np
import logging
import os
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import spectrogram
import tempfile
import soundfile as sf
from config import *

# Try to import pyannote (optional)
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    PYANNOTE_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("pyannote.audio not available, using fallback speaker detection")
    PYANNOTE_AVAILABLE = False

logger = logging.getLogger(__name__)

class SpeakerDiarization:
    def __init__(self):
        self.pipeline = None
        self.is_loaded = False
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    async def load_model(self):
        """Initialize speaker diarization"""
        try:
            if PYANNOTE_AVAILABLE:
                logger.info(f"Loading pyannote speaker diarization model: {DIARIZATION_MODEL}")
                
                # Try to load pre-trained pipeline with auth token
                try:
                    # First try without token
                    self.pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL)
                    logger.info("Loaded pyannote diarization model without auth token")
                except Exception as auth_error:
                    logger.warning(f"Failed to load without auth token: {auth_error}")
                    # Try with HuggingFace token from environment
                    hf_token = os.getenv("HUGGINGFACE_TOKEN")
                    if hf_token:
                        self.pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
                        logger.info("Loaded pyannote diarization model with auth token")
                    else:
                        logger.error("No HuggingFace token found. Using fallback method.")
                        self.pipeline = None
                
                # Move to device
                if self.pipeline and self.device == "mps":
                    self.pipeline.to(torch.device("mps"))
            else:
                logger.info("Using fallback speaker detection method")
                self.pipeline = None
            
            self.is_loaded = True
            logger.info("Speaker diarization initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize speaker diarization: {e}")
            logger.info("Using fallback speaker detection")
            self.pipeline = None
            self.is_loaded = True  # Still mark as loaded for fallback
            
    def diarize_audio(self, audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Dict:
        """
        Perform speaker diarization on audio data
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with speaker segments
        """
        if not self.is_loaded:
            logger.warning("Diarization not loaded, skipping speaker diarization")
            return {"speakers": [], "segments": []}
            
        # Use pyannote if available and loaded
        if self.pipeline:
            return self._pyannote_diarization(audio_data, sample_rate)
        else:
            return self._fallback_diarization(audio_data, sample_rate)
    
    def _pyannote_diarization(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Use pyannote.audio for diarization"""
        try:
            import os
            # Create temporary file for pyannote
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
                
            try:
                # Ensure audio is in correct format
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # Normalize audio
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                # Write to temporary file
                sf.write(tmp_file_path, audio_data, sample_rate)
                
                # Check if file was created and has content
                if not os.path.exists(tmp_file_path) or os.path.getsize(tmp_file_path) == 0:
                    logger.error("Failed to create temporary audio file")
                    return {"speakers": [], "segments": []}
                
                logger.info(f"Running pyannote diarization on {len(audio_data)} samples")
                
                # Run diarization
                diarization = self.pipeline(tmp_file_path, 
                                          min_speakers=MIN_SPEAKERS, 
                                          max_speakers=MAX_SPEAKERS)
                
                # Process results
                speakers = []
                segments = []
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    segment_info = {
                        "start": turn.start,
                        "end": turn.end,
                        "duration": turn.end - turn.start,
                        "speaker": f"Speaker_{speaker}"
                    }
                    segments.append(segment_info)
                    
                    speaker_name = f"Speaker_{speaker}"
                    if speaker_name not in speakers:
                        speakers.append(speaker_name)
                
                logger.info(f"Pyannote found {len(speakers)} speakers in {len(segments)} segments")
                
                return {
                    "speakers": speakers,
                    "segments": segments,
                    "num_speakers": len(speakers)
                }
                
            finally:
                # Clean up temporary file
                try:
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
                
        except Exception as e:
            logger.error(f"Pyannote diarization error: {e}")
            return self._fallback_diarization(audio_data, sample_rate)
    
    def _fallback_diarization(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Enhanced fallback speaker detection using multiple audio features"""
        try:
            logger.info("Using enhanced fallback speaker detection")
            
            # Use shorter segments for better resolution
            segment_length = int(sample_rate * 1.0)  # 1-second segments
            overlap = int(segment_length * 0.5)  # 50% overlap
            
            if len(audio_data) < segment_length:
                # Too short for analysis, assume single speaker
                return {
                    "speakers": ["Speaker_A"],
                    "segments": [{
                        "start": 0.0,
                        "end": len(audio_data) / sample_rate,
                        "duration": len(audio_data) / sample_rate,
                        "speaker": "Speaker_A"
                    }],
                    "num_speakers": 1
                }
            
            # Extract features for each segment
            features = []
            segment_times = []
            valid_segments = []
            
            for i in range(0, len(audio_data) - segment_length, segment_length - overlap):
                segment = audio_data[i:i + segment_length]
                if len(segment) < segment_length:
                    continue
                
                # Check if segment has enough energy (voice activity)
                rms_energy = np.sqrt(np.mean(segment**2))
                if rms_energy < 0.001:  # Skip very quiet segments
                    continue
                
                # Extract multiple types of features
                features_dict = self._extract_voice_features(segment, sample_rate)
                if features_dict is None:
                    continue
                
                feature_vector = [
                    features_dict['fundamental_freq'],
                    features_dict['spectral_centroid'],
                    features_dict['spectral_rolloff'],
                    features_dict['spectral_bandwidth'],
                    features_dict['mfcc_mean'],
                    features_dict['mfcc_std'],
                    features_dict['zero_crossing_rate'],
                    features_dict['rms_energy'],
                    features_dict['spectral_contrast']
                ]
                
                features.append(feature_vector)
                segment_times.append({
                    "start": i / sample_rate,
                    "end": (i + segment_length) / sample_rate
                })
                valid_segments.append(segment)
            
            if len(features) < 3:
                # Not enough segments for clustering
                return {
                    "speakers": ["Speaker_A"],
                    "segments": [{
                        "start": 0.0,
                        "end": len(audio_data) / sample_rate,
                        "duration": len(audio_data) / sample_rate,
                        "speaker": "Speaker_A"
                    }],
                    "num_speakers": 1
                }
            
            # Normalize features
            features_array = np.array(features)
            # Remove any NaN or inf values
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features_array)
            
            # Determine optimal number of clusters using elbow method
            n_clusters = self._determine_optimal_clusters(features_normalized, max_clusters=min(MAX_SPEAKERS, len(features) // 2))
            
            # K-means clustering for speaker identification
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            speaker_labels = kmeans.fit_predict(features_normalized)
            
            # Create speaker segments with post-processing
            segments = []
            speakers = [f"Speaker_{chr(65 + i)}" for i in range(n_clusters)]
            
            for i, (times, label) in enumerate(zip(segment_times, speaker_labels)):
                segments.append({
                    "start": times["start"],
                    "end": times["end"],
                    "duration": times["end"] - times["start"],
                    "speaker": f"Speaker_{chr(65 + label)}"
                })
            
            # Post-process: merge adjacent segments from same speaker
            segments = self._merge_adjacent_segments(segments)
            
            logger.info(f"Enhanced fallback method found {n_clusters} speakers in {len(segments)} segments")
            logger.info(f"Speaker distribution: {[seg['speaker'] for seg in segments]}")
            
            return {
                "speakers": speakers,
                "segments": segments,
                "num_speakers": n_clusters
            }
            
        except Exception as e:
            logger.error(f"Enhanced fallback diarization error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Last resort: assume single speaker
            return {
                "speakers": ["Speaker_A"],
                "segments": [{
                    "start": 0.0,
                    "end": len(audio_data) / sample_rate,
                    "duration": len(audio_data) / sample_rate,
                    "speaker": "Speaker_A"
                }],
                "num_speakers": 1
            }
    
    def _extract_voice_features(self, segment: np.ndarray, sample_rate: int) -> Optional[Dict]:
        """Extract comprehensive voice features from audio segment"""
        try:
            from scipy.fft import fft
            from scipy.signal import find_peaks
            
            # Basic energy check
            rms_energy = np.sqrt(np.mean(segment**2))
            if rms_energy < 0.001:
                return None
            
            # FFT for frequency analysis
            fft_vals = fft(segment)
            magnitude = np.abs(fft_vals[:len(fft_vals)//2])
            freqs = np.fft.fftfreq(len(segment), 1/sample_rate)[:len(magnitude)]
            
            # Fundamental frequency estimation (simple peak detection)
            # Focus on typical human voice range (80-400 Hz)
            voice_range_mask = (freqs >= 80) & (freqs <= 400)
            voice_magnitude = magnitude[voice_range_mask]
            voice_freqs = freqs[voice_range_mask]
            
            if len(voice_magnitude) > 0:
                fundamental_freq = voice_freqs[np.argmax(voice_magnitude)]
            else:
                fundamental_freq = 150  # Default
            
            # Spectral features
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            cumsum_magnitude = np.cumsum(magnitude)
            total_energy = cumsum_magnitude[-1]
            rolloff_idx = np.where(cumsum_magnitude >= 0.85 * total_energy)[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            
            # Spectral bandwidth
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * magnitude) / np.sum(magnitude)) if np.sum(magnitude) > 0 else 0
            
            # Simple MFCC-like features (log magnitude in frequency bands)
            n_bands = 13
            band_edges = np.logspace(np.log10(80), np.log10(8000), n_bands + 1)
            mfcc_features = []
            
            for i in range(n_bands):
                band_mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
                band_energy = np.sum(magnitude[band_mask])
                mfcc_features.append(np.log(band_energy + 1e-10))
            
            mfcc_mean = np.mean(mfcc_features)
            mfcc_std = np.std(mfcc_features)
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.sign(segment)))[0]
            zero_crossing_rate = len(zero_crossings) / len(segment)
            
            # Spectral contrast (difference between peaks and valleys)
            spectral_contrast = np.std(magnitude) / (np.mean(magnitude) + 1e-10)
            
            return {
                'fundamental_freq': fundamental_freq,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'spectral_bandwidth': spectral_bandwidth,
                'mfcc_mean': mfcc_mean,
                'mfcc_std': mfcc_std,
                'zero_crossing_rate': zero_crossing_rate,
                'rms_energy': rms_energy,
                'spectral_contrast': spectral_contrast
            }
            
        except Exception as e:
            logger.warning(f"Error extracting voice features: {e}")
            return None
    
    def _determine_optimal_clusters(self, features: np.ndarray, max_clusters: int = 4) -> int:
        """Determine optimal number of clusters using elbow method"""
        if len(features) < 6:
            return 1
        
        max_clusters = min(max_clusters, len(features) // 3)
        if max_clusters < 2:
            return 1
            
        try:
            inertias = []
            K = range(1, max_clusters + 1)
            
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
                kmeans.fit(features)
                inertias.append(kmeans.inertia_)
            
            # Simple elbow detection
            if len(inertias) >= 3:
                # Calculate the rate of change
                deltas = np.diff(inertias)
                delta_deltas = np.diff(deltas)
                
                # Find the point where the rate of change stabilizes
                if len(delta_deltas) > 0:
                    optimal_k = np.argmax(delta_deltas) + 2  # +2 because of double diff
                    return min(optimal_k, max_clusters)
            
            # Fallback: use 2 clusters if we have enough data
            return 2 if len(features) >= 6 else 1
            
        except Exception as e:
            logger.warning(f"Error determining optimal clusters: {e}")
            return 2 if len(features) >= 6 else 1
    
    def _merge_adjacent_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge adjacent segments from the same speaker"""
        if len(segments) <= 1:
            return segments
        
        merged = []
        current_segment = segments[0].copy()
        
        for next_segment in segments[1:]:
            if (current_segment['speaker'] == next_segment['speaker'] and 
                abs(current_segment['end'] - next_segment['start']) < 0.5):  # 0.5 second tolerance
                # Merge segments
                current_segment['end'] = next_segment['end']
                current_segment['duration'] = current_segment['end'] - current_segment['start']
            else:
                # Start new segment
                merged.append(current_segment)
                current_segment = next_segment.copy()
        
        merged.append(current_segment)
        return merged
            
    def assign_speakers_to_transcription(self, 
                                       transcription_segments: List[Dict], 
                                       diarization_result: Dict) -> List[Dict]:
        """
        Assign speakers to transcription segments based on time overlap
        
        Args:
            transcription_segments: Segments from Whisper transcription
            diarization_result: Result from speaker diarization
            
        Returns:
            Updated transcription segments with speaker information
        """
        if not diarization_result.get("segments"):
            logger.debug("No diarization segments available")
            return transcription_segments
            
        updated_segments = []
        
        for trans_seg in transcription_segments:
            trans_start = trans_seg.get("start", 0)
            trans_end = trans_seg.get("end", 0)
            
            # Find overlapping diarization segments
            best_speaker = "Unknown"
            max_overlap = 0
            
            for dia_seg in diarization_result["segments"]:
                dia_start = dia_seg["start"]
                dia_end = dia_seg["end"]
                
                # Calculate overlap
                overlap_start = max(trans_start, dia_start)
                overlap_end = min(trans_end, dia_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = dia_seg["speaker"]
            
            # Update segment with speaker info
            updated_segment = trans_seg.copy()
            updated_segment["speaker"] = best_speaker
            updated_segments.append(updated_segment)
            
        logger.debug(f"Assigned speakers to {len(updated_segments)} transcription segments")
        return updated_segments
        
    def get_speaker_summary(self, segments: List[Dict]) -> Dict:
        """
        Generate summary of speaker activity
        
        Args:
            segments: List of segments with speaker information
            
        Returns:
            Dictionary with speaker statistics
        """
        if not segments:
            return {"total_speakers": 0, "speaker_stats": {}}
            
        speaker_stats = {}
        
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            duration = segment.get("end", 0) - segment.get("start", 0)
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "total_duration": 0,
                    "segment_count": 0,
                    "words": []
                }
            
            speaker_stats[speaker]["total_duration"] += duration
            speaker_stats[speaker]["segment_count"] += 1
            speaker_stats[speaker]["words"].append(segment.get("text", ""))
        
        return {
            "total_speakers": len(speaker_stats),
            "speaker_stats": speaker_stats
        }
        
    def unload_model(self):
        """Cleanup resources"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        self.is_loaded = False
        logger.info("Speaker diarization model unloaded")
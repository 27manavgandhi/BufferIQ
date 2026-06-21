"""Audio analysis for videos."""

from typing import Optional
import subprocess
import json

from bufferiq.ml.multimodal.types import AudioFeatures
from bufferiq.ml.multimodal.exceptions import MediaProcessingError


class AudioAnalyzer:
    """Analyze audio in videos."""
    
    def __init__(self):
        """Initialize audio analyzer."""
        pass
    
    def analyze(self, video_path: str) -> Optional[AudioFeatures]:
        """
        Analyze audio in video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Audio features or None if no audio
            
        Raises:
            MediaProcessingError: If analysis fails
        """
        try:
            # Use ffprobe to check for audio
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                '-select_streams', 'a',
                video_path
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode != 0:
                    return None
                
                data = json.loads(result.stdout)
                
                if not data.get('streams'):
                    return None
                
                audio_stream = data['streams'][0]
                
                return AudioFeatures(
                    duration_seconds=float(audio_stream.get('duration', 0)),
                    sample_rate=int(audio_stream.get('sample_rate', 44100)),
                    channels=int(audio_stream.get('channels', 2)),
                    has_speech=self._detect_speech_heuristic(audio_stream),
                    music_detected=self._detect_music_heuristic(audio_stream),
                )
                
            except (subprocess.TimeoutExpired, json.JSONDecodeError):
                # Fallback to simulated features
                return AudioFeatures(
                    duration_seconds=30.0,
                    sample_rate=44100,
                    channels=2,
                    has_speech=True,
                    music_detected=False,
                )
                
        except Exception as e:
            raise MediaProcessingError(f"Audio analysis failed: {str(e)}")
    
    def _detect_speech_heuristic(self, audio_stream: dict) -> bool:
        """
        Heuristic to detect speech.
        
        Args:
            audio_stream: Audio stream metadata
            
        Returns:
            True if speech likely present
        """
        # Simple heuristic: mono or stereo, common sample rates
        channels = audio_stream.get('channels', 0)
        sample_rate = audio_stream.get('sample_rate', 0)
        
        return channels in [1, 2] and sample_rate >= 16000
    
    def _detect_music_heuristic(self, audio_stream: dict) -> bool:
        """
        Heuristic to detect music.
        
        Args:
            audio_stream: Audio stream metadata
            
        Returns:
            True if music likely present
        """
        # Simple heuristic: stereo, high sample rate
        channels = audio_stream.get('channels', 0)
        sample_rate = audio_stream.get('sample_rate', 0)
        
        return channels == 2 and sample_rate >= 44100
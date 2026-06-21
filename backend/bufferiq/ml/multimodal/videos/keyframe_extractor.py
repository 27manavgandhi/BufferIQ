"""Key frame extraction from videos."""

from typing import List
import cv2
import numpy as np

from bufferiq.ml.multimodal.types import KeyFrame
from bufferiq.ml.multimodal.exceptions import MediaProcessingError


class KeyFrameExtractor:
    """Extract key frames from videos."""
    
    def __init__(self, threshold: float = 30.0):
        """
        Initialize key frame extractor.
        
        Args:
            threshold: Difference threshold for key frame detection
        """
        self.threshold = threshold
    
    def extract(
        self,
        video_path: str,
        max_frames: int = 10
    ) -> List[KeyFrame]:
        """
        Extract key frames from video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of key frames to extract
            
        Returns:
            List of key frames
            
        Raises:
            MediaProcessingError: If extraction fails
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise MediaProcessingError(f"Cannot open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            keyframes = []
            
            prev_frame = None
            frame_idx = 0
            
            while len(keyframes) < max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Check if this is a key frame
                if prev_frame is not None:
                    # Calculate difference from previous frame
                    diff = cv2.absdiff(prev_frame, gray)
                    diff_score = np.mean(diff)
                    
                    if diff_score > self.threshold:
                        # This is a key frame
                        timestamp = frame_idx / fps if fps > 0 else 0
                        
                        keyframe = KeyFrame(
                            timestamp=timestamp,
                            frame_index=frame_idx,
                            thumbnail_url=f"/tmp/keyframe_{frame_idx}.jpg",
                            importance_score=float(diff_score / 255.0),
                        )
                        keyframes.append(keyframe)
                
                prev_frame = gray
                frame_idx += 1
            
            cap.release()
            
            return keyframes
            
        except Exception as e:
            raise MediaProcessingError(f"Key frame extraction failed: {str(e)}")
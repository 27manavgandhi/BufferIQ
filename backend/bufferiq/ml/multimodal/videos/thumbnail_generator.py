"""Video thumbnail generation."""

from typing import List
import cv2
import numpy as np
from PIL import Image

from bufferiq.ml.multimodal.exceptions import MediaProcessingError


class ThumbnailGenerator:
    """Generate thumbnails from videos."""
    
    def __init__(self, output_dir: str = "/tmp/thumbnails"):
        """
        Initialize thumbnail generator.
        
        Args:
            output_dir: Directory to save thumbnails
        """
        self.output_dir = output_dir
    
    def generate(
        self,
        video_path: str,
        count: int = 3,
        timestamps: List[float] | None = None
    ) -> List[str]:
        """
        Generate thumbnails from video.
        
        Args:
            video_path: Path to video file
            count: Number of thumbnails to generate
            timestamps: Specific timestamps (seconds) to extract
            
        Returns:
            List of thumbnail file paths
            
        Raises:
            MediaProcessingError: If generation fails
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise MediaProcessingError(f"Cannot open video: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Determine timestamps
            if timestamps is None:
                # Generate evenly spaced timestamps
                timestamps = [
                    duration * (i + 1) / (count + 1)
                    for i in range(count)
                ]
            
            thumbnail_paths = []
            
            for idx, timestamp in enumerate(timestamps):
                # Seek to timestamp
                frame_number = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                # Read frame
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    image = Image.fromarray(frame_rgb)
                    
                    # Save thumbnail
                    thumbnail_path = f"{self.output_dir}/thumb_{idx}.jpg"
                    image.save(thumbnail_path, quality=85)
                    thumbnail_paths.append(thumbnail_path)
            
            cap.release()
            
            return thumbnail_paths
            
        except Exception as e:
            raise MediaProcessingError(f"Thumbnail generation failed: {str(e)}")
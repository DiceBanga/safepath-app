"""
SafePath Inference Pipeline
==========================

Real-time inference pipeline for hazard detection on mobile devices.
"""

import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import cv2
import numpy as np

from src.models.deeplabv3plus import DeepLabV3Plus


class InferencePipeline:
    """
    Real-time inference pipeline for SafePath hazard detection.
    
    Handles:
    - Camera feed capture
    - Preprocessing (low-light enhancement)
    - Model inference
    - Post-processing (hazard classification, overlay)
    - Alert generation
    - Report logging
    
    Args:
        model: Trained DeepLabV3+ model
        device: Inference device ('cpu', 'cuda', 'npu')
        confidence_threshold: Minimum confidence for hazard detection
        target_fps: Target frames per second
    """
    
    # Color mapping for hazard overlay
    HAZARD_COLORS = {
        'pothole': (255, 0, 0),      # Red
        'pole': (255, 165, 0),       # Orange
        'uneven_terrain': (255, 255, 0),  # Yellow
        'water_puddle': (0, 100, 255),    # Blue
        'safe_path': (0, 255, 0),    # Green
        'road': (128, 128, 128),     # Gray
        'sidewalk': (192, 192, 192), # Light Gray
    }
    
    def __init__(
        self,
        model: DeepLabV3Plus,
        device: str = 'cpu',
        confidence_threshold: float = 0.7,
        target_fps: int = 5
    ):
        self.model = model.to(device)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.target_fps = target_fps
        
        self.model.eval()
        
        # Detection logging
        self.detection_log: List[Dict] = []
        self.frame_count = 0
        self.start_time = None
        
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for inference.
        
        Args:
            frame: Input BGR frame from camera
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        frame_resized = cv2.resize(frame_rgb, (960, 480))
        
        # Normalize
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_normalized = (frame_normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # Convert to tensor
        tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def infer(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Run inference on preprocessed tensor.
        
        Args:
            tensor: Preprocessed input tensor
            
        Returns:
            Segmentation mask as numpy array
        """
        with torch.no_grad():
            output = self.model(tensor)
            mask = output.argmax(dim=1).squeeze().cpu().numpy()
            
        return mask
    
    def postprocess(
        self, 
        mask: np.ndarray, 
        original_frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Post-process segmentation mask.
        
        Args:
            mask: Segmentation mask from model
            original_frame: Original camera frame
            
        Returns:
            Tuple of (overlay frame, detected hazards)
        """
        # Create color overlay
        overlay = original_frame.copy()
        hazards_detected = []
        
        # Map class IDs to colors and detect hazards
        for class_name, color in self.HAZARD_COLORS.items():
            # Get class ID from model output
            # This would be implemented based on actual class mapping
            
            # For demonstration:
            if class_name in ['pothole', 'pole', 'uneven_terrain', 'water_puddle']:
                # These are hazard classes - would trigger alerts
                hazards_detected.append({
                    'type': class_name,
                    'confidence': 0.85,  # Placeholder
                    'location': 'center',  # Placeholder
                    'timestamp': time.time()
                })
        
        return overlay, hazards_detected
    
    def generate_alert(self, hazards: List[Dict]) -> Optional[str]:
        """
        Generate alert message based on detected hazards.
        
        Args:
            hazards: List of detected hazards
            
        Returns:
            Alert message or None
        """
        if not hazards:
            return None
            
        # Prioritize hazards
        priority_order = ['pothole', 'uneven_terrain', 'pole', 'water_puddle']
        
        for hazard_type in priority_order:
            for hazard in hazards:
                if hazard['type'] == hazard_type:
                    return f"Warning: {hazard_type.replace('_', ' ')} detected ahead!"
        
        return "Caution: Potential hazard detected"
    
    def log_detection(self, hazards: List[Dict], frame: np.ndarray):
        """
        Log detection for report generation.
        
        Args:
            hazards: Detected hazards
            frame: Current frame
        """
        for hazard in hazards:
            self.detection_log.append({
                **hazard,
                'frame_id': self.frame_count
            })
    
    def run_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Tuple of (processed frame with overlay, alert message)
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        # Preprocess
        tensor = self.preprocess(frame)
        
        # Inference
        mask = self.infer(tensor)
        
        # Postprocess
        overlay, hazards = self.postprocess(mask, frame)
        
        # Generate alert
        alert = self.generate_alert(hazards)
        
        # Log detection
        if hazards:
            self.log_detection(hazards, frame)
        
        self.frame_count += 1
        
        return overlay, alert
    
    def get_stats(self) -> Dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with FPS, total frames, detections
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return {
            'fps': fps,
            'total_frames': self.frame_count,
            'total_detections': len(self.detection_log),
            'elapsed_time': elapsed
        }
    
    def reset(self):
        """Reset pipeline state for new session."""
        self.detection_log = []
        self.frame_count = 0
        self.start_time = None


class CameraManager:
    """
    Manages camera feed for inference pipeline.
    """
    
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (960, 480)):
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap = None
        
    def start(self):
        """Initialize camera capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from camera."""
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def stop(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


if __name__ == "__main__":
    # Test inference pipeline
    print("Testing InferencePipeline...")
    
    # Create model and pipeline
    # model = DeepLabV3Plus(num_classes=19)
    # pipeline = InferencePipeline(model, device='cpu')
    
    # Test with dummy frame
    # dummy_frame = np.random.randint(0, 255, (480, 960, 3), dtype=np.uint8)
    # overlay, alert = pipeline.run_frame(dummy_frame)
    # print(f"Alert: {alert}")
    # print(f"Stats: {pipeline.get_stats()}")

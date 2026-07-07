"""
Phase 2: Face Detection & Alignment Module
Detects faces using OpenCV DNN and aligns so eyes are perfectly horizontal.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import sys
import os
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contracts import FrameData, FaceData


class FaceModule:
    """
    Face detection and alignment using OpenCV's DNN face detector
    plus geometric eye estimation for affine alignment.
    """
    
    # Model URLs and paths
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (160, 160),
                 detection_threshold: float = 0.5):
        """
        Args:
            target_size: (width, height) for aligned face crop
            detection_threshold: Minimum confidence for face detection [0.0, 1.0]
        """
        self.target_size = target_size
        self.detection_threshold = detection_threshold
        
        # Ensure models exist
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        self._download_models_if_needed()
        
        # Load DNN face detector
        self.net = cv2.dnn.readNetFromCaffe(
            os.path.join(self.MODEL_DIR, "deploy.prototxt"),
            os.path.join(self.MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        )
        
        print("[FaceModule] Using OpenCV DNN face detector (SSD)")
        print(f"[FaceModule] Target crop size: {target_size}")
        print(f"[FaceModule] Detection threshold: {detection_threshold}")
    
    def _download_models_if_needed(self):
        """Download DNN model files if not present."""
        prototxt_path = os.path.join(self.MODEL_DIR, "deploy.prototxt")
        caffemodel_path = os.path.join(self.MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        
        if not os.path.exists(prototxt_path):
            print("[FaceModule] Downloading deploy.prototxt...")
            urllib.request.urlretrieve(self.PROTOTXT_URL, prototxt_path)
        
        if not os.path.exists(caffemodel_path):
            print("[FaceModule] Downloading caffemodel (this may take a moment)...")
            urllib.request.urlretrieve(self.CAFFEMODEL_URL, caffemodel_path)
    
    # ──────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────
    
    def detect_faces(self, frame_data: FrameData) -> List[FaceData]:
        """
        Detect all faces in a frame and return aligned crops.
        
        Args:
            frame_data: The captured frame
            
        Returns:
            List of FaceData objects (one per detected face)
        """
        frame = frame_data.pixel_array
        h, w = frame.shape[:2]
        
        # Prepare blob for DNN
        blob = cv2.dnn.blobFromImage(
            frame, 
            scalefactor=1.0, 
            size=(300, 300), 
            mean=(104.0, 177.0, 123.0)
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence < self.detection_threshold:
                continue
            
            # Get bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            
            # Ensure within frame bounds
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            bbox_w = x2 - x
            bbox_h = y2 - y
            
            if bbox_w <= 0 or bbox_h <= 0:
                continue
            
            # Estimate eye positions
            left_eye, right_eye = self._estimate_eyes(x, y, bbox_w, bbox_h)
            
            # Align face
            aligned_crop = self._align_face(frame, left_eye, right_eye)
            
            if aligned_crop is not None:
                face_data = FaceData(
                    cropped_face=aligned_crop,
                    bbox=(int(x), int(y), int(bbox_w), int(bbox_h)),
                    left_eye=left_eye,
                    right_eye=right_eye,
                    confidence=float(confidence),
                    source_frame_number=frame_data.frame_number,
                    target_size=self.target_size
                )
                faces.append(face_data)
        
        return faces
    
    def detect_largest_face(self, frame_data: FrameData) -> Optional[FaceData]:
        """
        Detect only the largest face in frame (typically the subject).
        """
        faces = self.detect_faces(frame_data)
        
        if not faces:
            return None
        
        return max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    
    # ──────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────
    
    def _estimate_eyes(self, x: int, y: int, w: int, h: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Estimate eye positions from face bounding box using anatomical ratios.
        
        In a typical face:
        - Eyes are ~25-35% down from the top of the bounding box
        - Left eye is ~25-30% from the left edge
        - Right eye is ~70-75% from the left edge
        """
        # Left eye: ~28% from left, ~30% from top
        left_eye_x = x + int(w * 0.28)
        left_eye_y = y + int(h * 0.30)
        
        # Right eye: ~72% from left, ~30% from top
        right_eye_x = x + int(w * 0.72)
        right_eye_y = y + int(h * 0.30)
        
        return (left_eye_x, left_eye_y), (right_eye_x, right_eye_y)
    
    def _align_face(self, frame: np.ndarray, 
                    left_eye: Tuple[int, int], 
                    right_eye: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Apply affine transformation so eyes are horizontal and centered.
        """
        target_w, target_h = self.target_size
        
        # Desired eye positions in the output crop
        desired_left_eye = (
            int(target_w * 0.30),
            int(target_h * 0.40)
        )
        desired_right_eye = (
            int(target_w * 0.70),
            int(target_h * 0.40)
        )
        
        # Compute angle between eyes
        dx = float(right_eye[0] - left_eye[0])
        dy = float(right_eye[1] - left_eye[1])
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Compute scaling factor based on eye distance
        eye_distance = np.sqrt(dx**2 + dy**2)
        desired_distance = np.sqrt(
            (desired_right_eye[0] - desired_left_eye[0])**2 +
            (desired_right_eye[1] - desired_left_eye[1])**2
        )
        
        if eye_distance < 1.0:
            return None
        
        scale = desired_distance / eye_distance
        
        # Center point between the eyes
        eye_center_x = float(left_eye[0] + right_eye[0]) / 2.0
        eye_center_y = float(left_eye[1] + right_eye[1]) / 2.0
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D((eye_center_x, eye_center_y), angle, scale)
        
        # Adjust translation so eyes land at desired positions
        M[0, 2] += desired_left_eye[0] - eye_center_x
        M[1, 2] += desired_left_eye[1] - eye_center_y
        
        # Apply affine transformation
        aligned = cv2.warpAffine(
            frame, M, (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return aligned
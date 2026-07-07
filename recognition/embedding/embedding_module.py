"""
Phase 3: Face Embedding Module
Converts aligned face crops into 512-dim L2-normalized vectors.
Uses FaceNet (InceptionResnetV1) from facenet-pytorch.
"""

import numpy as np
from typing import Optional
import sys
import os
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contracts import FaceData, EmbeddingData


class EmbeddingModule:
    """
    Face embedding extraction using FaceNet (InceptionResnetV1).
    
    Input: Aligned face crop (BGR, 160x160)
    Output: L2-normalized 512-dimensional embedding vector
    """
    
    def __init__(self, model_name: str = "vggface2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._input_size = (160, 160)
        self.resnet = None
        
        self._load_model()
    
    def _load_model(self):
        """Load FaceNet model."""
        try:
            from facenet_pytorch import InceptionResnetV1
            import torch
            
            print(f"[EmbeddingModule] Loading FaceNet ({self.model_name}) on {self.device}...")
            
            self.resnet = InceptionResnetV1(
                pretrained=self.model_name,
                classify=False,
                device=self.device
            ).eval()
            
            print(f"[EmbeddingModule] Model loaded successfully")
            print(f"[EmbeddingModule] Input size: {self._input_size}")
            print(f"[EmbeddingModule] Output dimension: 512")
            
        except ImportError:
            print("[EmbeddingModule] ERROR: facenet-pytorch not installed.")
            print("[EmbeddingModule] Install with: pip install facenet-pytorch torch")
            raise
        except Exception as e:
            print(f"[EmbeddingModule] ERROR loading model: {e}")
            raise
    
    def extract(self, face_data: FaceData) -> Optional[EmbeddingData]:
        """Extract embedding from an aligned face crop."""
        import torch
        import torch.nn.functional as F
        
        start_time = time.time()
        
        try:
            face_crop = face_data.cropped_face
            
            # BGR -> RGB
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Resize to 160x160 if needed
            if face_rgb.shape[:2] != self._input_size:
                face_rgb = cv2.resize(face_rgb, self._input_size)
            
            # Convert to tensor and normalize to [-1, 1]
            face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
            face_tensor = (face_tensor - 127.5) / 128.0
            face_tensor = face_tensor.unsqueeze(0)
            
            if self.device == "cuda":
                face_tensor = face_tensor.cuda()
            
            with torch.no_grad():
                embedding = self.resnet(face_tensor)
                embedding = F.normalize(embedding, p=2, dim=1)
            
            vector = embedding.cpu().numpy().flatten()
            extraction_time = (time.time() - start_time) * 1000
            
            return EmbeddingData(
                vector=vector,
                dimension=len(vector),
                source_frame_number=face_data.source_frame_number,
                model_name=self.model_name,
                extraction_time_ms=round(extraction_time, 2)
            )
            
        except Exception as e:
            print(f"[EmbeddingModule] Extraction failed: {e}")
            return None
    
    def extract_batch(self, face_data_list: list) -> list:
        return [self.extract(face) for face in face_data_list]
    
    def compare(self, emb1: EmbeddingData, emb2: EmbeddingData) -> dict:
        similarity = emb1.cosine_similarity(emb2)
        distance = emb1.cosine_distance(emb2)
        return {
            'similarity': round(similarity, 6),
            'distance': round(distance, 6),
            'same_person': similarity > 0.5
        }
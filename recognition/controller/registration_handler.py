"""
Registration Handler — Phase 7 Enhanced

"""

import time
import numpy as np
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple

from camera.camera_module import CameraModule
from face.face_module import FaceModule
from face.liveness import LivenessGate
from embedding.embedding_module import EmbeddingModule
from database.database_module import DatabaseModule
from contracts import (
    FrameData, FaceData, EmbeddingData,
    StorageResult, RegistrationResult,
    QualityMetadata, EnrollmentQuality,
)


class RegistrationHandler:
    """
    Handles synchronous face registration with quality-weighted enrollment.
    
    The camera is engaged only during capture_frames().
    No background threads — caller invokes register() and waits for result.
    """
    
    # ──────────────────────────────────────────────
    # CONFIGURATION
    # ──────────────────────────────────────────────
    
    FRAME_COUNT = 5                     # Frames to attempt
    FRAME_INTERVAL = 0.6                # Seconds between frames (~3s total)
    BUFFER_MULTIPLIER = 2               # Capture 2× frames, select best 5
    
    # Duplicate check thresholds
    DUPLICATE_THRESHOLD_CENTROID = 0.30  # Centroid vs DB
    DUPLICATE_THRESHOLD_BEST_FRAME = 0.25  # Best single frame vs DB
    DUPLICATE_THRESHOLD_LOW_QUALITY = 0.35  # Relaxed when quality is poor
    MIN_FRAMES_FOR_DUPLICATE_CHECK = 2
    
    # Quality thresholds
    QUALITY_GRADE_EXCELLENT = 0.85
    QUALITY_GRADE_GOOD = 0.70
    QUALITY_GRADE_MARGINAL = 0.50
    # Below 0.50 = "poor"
    
    # Lighting thresholds (ratio of variance to liveness threshold)
    LIGHTING_EXCELLENT = 2.0    # Variance > 2× threshold
    LIGHTING_GOOD = 1.2         # Variance > 1.2× threshold
    LIGHTING_ACCEPTABLE = 0.8   # Variance > 0.8× threshold
    LIGHTING_LOW = 0.5          # Variance > 0.5× threshold
    # Below 0.5 = "insufficient"
    
    # Correlation gate for re-enrollment
    RE_ENROLL_CORRELATION_MIN = 0.70  # Old vs new centroid similarity
    RE_ENROLL_BLEND_RATIO = 0.30      # Keep 30% old, 70% new
    
    def __init__(self,
                 camera: CameraModule,
                 face_detector: FaceModule,
                 liveness_gate: LivenessGate,
                 embedder: EmbeddingModule,
                 database: DatabaseModule):
        self.camera = camera
        self.face_detector = face_detector
        self.liveness_gate = liveness_gate
        self.embedder = embedder
        self.database = database
    
    # ──────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────
    
    def register(self, user_id: str, name: str,
                 metadata: Optional[Dict] = None) -> RegistrationResult:
        """
        Execute the full registration flow synchronously.

        """
        start_time = time.time()
        metadata = metadata or {}
        
        # Step 1: Capture frames
        frames = self._capture_frames()
        if len(frames) < 2:
            return self._fail(
                user_id, name, start_time,
                f"Insufficient frames captured ({len(frames)}/{self.FRAME_COUNT}). "
                f"Ensure face is visible and well-lit."
            )
        
        # Step 2: Process each frame → embeddings + quality metadata
        embeddings, quality_metadata_list, liveness_results = (
            self._process_frames_with_quality(frames)
        )
        
        if len(embeddings) == 0:
            return self._fail(
                user_id, name, start_time,
                "No valid embeddings extracted.",
                liveness_results[0] if liveness_results else None,
                quality=None
            )
        
        # Step 2b: Select the most diverse subset of frames
        embeddings, quality_metadata_list, diversity_score = (
            self._select_diverse(embeddings, quality_metadata_list)
        )
        
        if len(embeddings) < 2:
            return self._fail(
                user_id, name, start_time,
                f"Insufficient diverse frames ({len(embeddings)}). "
                f"Ensure face movement during capture.",
                self._summarize_liveness(liveness_results),
                quality=None
            )
        
        # Step 3: Compute quality-weighted centroid
        centroid_vector, weights = self._compute_weighted_centroid(
            embeddings, quality_metadata_list
        )
        
        # Update quality metadata with computed weights and self-similarity
        for i, qm in enumerate(quality_metadata_list):
            qm.weight = weights[i] if i < len(weights) else 0.0
            if i < len(embeddings):
                qm.embedding_quality = float(
                    np.dot(centroid_vector, embeddings[i].vector)
                )
        
        centroid_embedding = EmbeddingData(
            vector=centroid_vector,
            dimension=512,
            source_frame_number=-1,
            model_name="centroid_weighted",
            extraction_time_ms=0.0
        )
        
        # Step 4: Compute enrollment quality score
        quality = self._compute_enrollment_quality(
            quality_metadata_list, centroid_vector, embeddings,
            pose_diversity=diversity_score
        )
        
        # Step 5: Layered duplicate check
        if len(embeddings) >= self.MIN_FRAMES_FOR_DUPLICATE_CHECK:
            is_duplicate, duplicate_match = self._check_duplicate_layered(
                centroid_embedding, embeddings, quality_metadata_list, quality
            )
            if is_duplicate:
                return self._fail(
                    user_id, name, start_time,
                    f"Identity already exists: '{duplicate_match['name']}' "
                    f"(distance={duplicate_match['distance']:.4f})",
                    liveness=self._summarize_liveness(liveness_results),
                    quality=quality
                )
        else:
            print(f"[RegistrationHandler] Skipping duplicate check "
                  f"({len(embeddings)} embeddings < "
                  f"{self.MIN_FRAMES_FOR_DUPLICATE_CHECK} minimum)")
        
        # Step 6: Store in database with full enrollment metadata
        storage_result = self.database.store(
            user_id=user_id,
            name=name,
            embedding=centroid_embedding,
            metadata={
                **metadata,
                "frames_used": len(embeddings),
                "frames_captured": len(frames),
                "registration_method": "centroid_weighted",
                "quality_score": quality.overall_score,
                "quality_grade": quality.grade,
                "lighting_condition": quality.lighting_condition,
                "avg_self_similarity": quality.avg_self_similarity,
                "pose_diversity": quality.pose_diversity,
                "per_frame_count": len(quality_metadata_list),
                "per_frame_confidence": [qm.confidence for qm in quality_metadata_list],
                "per_frame_variance": [qm.liveness_variance for qm in quality_metadata_list],
                "per_frame_lighting": [qm.lighting for qm in quality_metadata_list],
                "per_frame_weight": [qm.weight for qm in quality_metadata_list],
                "per_frame_embedding_quality": [qm.embedding_quality for qm in quality_metadata_list],
            }
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if storage_result.verify_roundtrip():
            return RegistrationResult(
                status="success",
                transaction_id=storage_result.document_id,
                user_id=user_id,
                name=name,
                liveness=self._summarize_liveness(liveness_results),
                embedding_dim=512,
                processing_time_ms=round(processing_time, 2),
                timestamp=datetime.now(timezone.utc).isoformat(),
                quality={
                    "overall_score": quality.overall_score,
                    "grade": quality.grade,
                    "avg_confidence": quality.avg_confidence,
                    "avg_liveness_variance": quality.avg_liveness_variance,
                    "lighting_condition": quality.lighting_condition,
                    "pose_diversity": quality.pose_diversity,
                    "avg_self_similarity": quality.avg_self_similarity,
                    "frames_used": quality.frames_used,
                    "frames_captured": quality.frames_captured,
                },
                quality_score=quality.overall_score,
                quality_grade=quality.grade,
                lighting_condition=quality.lighting_condition,
                frames_used=quality.frames_used,
                frames_captured=quality.frames_captured,
                avg_self_similarity=quality.avg_self_similarity,
                recommendation=quality.recommendation,
            )
        else:
            return self._fail(
                user_id, name, start_time,
                "Database roundtrip verification failed",
                liveness=self._summarize_liveness(liveness_results),
                quality=quality
            )
    
    def re_enroll(self, user_id: str) -> RegistrationResult:
        """
        Update an existing identity with fresh face data.

        """
        start_time = time.time()
        
        # Gate 1: Verify existing identity
        existing = self.database.get_by_user_id(user_id)
        if not existing:
            return self._fail(
                user_id, "", start_time,
                f"No existing identity found for user_id '{user_id}'"
            )
        
        existing_vector = existing[0]['embedding']
        existing_name = existing[0]['name']
        existing_metadata = existing[0].get('metadata', {})
        
        # Capture and process new frames
        frames = self._capture_frames()
        if len(frames) < 2:
            return self._fail(
                user_id, existing_name, start_time,
                "Insufficient frames for re-enrollment"
            )
        
        embeddings, quality_metadata_list, liveness_results = (
            self._process_frames_with_quality(frames)
        )
        
        if len(embeddings) == 0:
            return self._fail(
                user_id, existing_name, start_time,
                "No valid embeddings in re-enrollment"
            )
        
        # Select diverse subset
        embeddings, quality_metadata_list, diversity_score = (
            self._select_diverse(embeddings, quality_metadata_list)
        )
        
        if len(embeddings) < 2:
            return self._fail(
                user_id, existing_name, start_time,
                "Insufficient diverse frames for re-enrollment"
            )
        
        # Compute new centroid
        new_centroid, _ = self._compute_weighted_centroid(
            embeddings, quality_metadata_list
        )
        
        # Gate 3: Correlation check
        correlation = float(np.dot(existing_vector, new_centroid))
        
        if correlation < self.RE_ENROLL_CORRELATION_MIN:
            return self._fail(
                user_id, existing_name, start_time,
                f"New face data doesn't correlate with stored identity "
                f"(similarity={correlation:.3f}, need >"
                f"{self.RE_ENROLL_CORRELATION_MIN}). "
                f"Delete and re-register if this is intentional."
            )
        
        # Gate 4: Blend old and new
        blend_ratio = self.RE_ENROLL_BLEND_RATIO
        blended = (
            (1.0 - blend_ratio) * existing_vector +
            blend_ratio * new_centroid
        )
        blended = blended / np.linalg.norm(blended)
        
        # Update database
        success, update_result = self.database.update_vector(
            existing[0]['id'], blended, ema_alpha=blend_ratio
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if success:
            quality = self._compute_enrollment_quality(
                quality_metadata_list, new_centroid, embeddings,
                pose_diversity=diversity_score
            )
            
            return RegistrationResult(
                status="success",
                transaction_id=existing[0]['id'],
                user_id=user_id,
                name=existing_name,
                liveness=self._summarize_liveness(liveness_results),
                embedding_dim=512,
                processing_time_ms=round(processing_time, 2),
                timestamp=datetime.now(timezone.utc).isoformat(),
                quality={
                    "overall_score": quality.overall_score,
                    "grade": quality.grade,
                    "correlation": round(correlation, 4),
                    "blend_ratio": blend_ratio,
                },
                quality_score=quality.overall_score,
                quality_grade=quality.grade,
                recommendation=f"Re-enrollment successful. "
                               f"Correlation: {correlation:.3f}. "
                               f"Blend: {blend_ratio:.0%} new data.",
            )
        else:
            return self._fail(
                user_id, existing_name, start_time,
                "Database update failed during re-enrollment"
            )
    
    # ──────────────────────────────────────────────────
    # INTERNAL: DIVERSE FRAME SELECTION
    # ──────────────────────────────────────────────────
    
    def _select_diverse(
        self,
        embeddings: List[EmbeddingData],
        quality_metadata_list: List[QualityMetadata],
    ) -> Tuple[List[EmbeddingData], List[QualityMetadata], float]:
        """
        Select the N most diverse frames from the candidate pool.
        
        Uses greedy farthest-point sampling on cosine distance.
        Returns the selected subset and the diversity score D.
        
        Diversity score D (0.0–1.0):
            1.0 = perfectly diverse (all mutually orthogonal)
            0.0 = all identical (clone attack)
        """
        n_select = min(self.FRAME_COUNT, len(embeddings))
        
        if n_select < 2 or len(embeddings) <= n_select:
            # Not enough to select from, or already at target
            if len(embeddings) >= 2:
                similarities = [
                    float(np.dot(embeddings[i].vector, embeddings[j].vector))
                    for i in range(len(embeddings))
                    for j in range(i + 1, len(embeddings))
                ]
                diversity = 1.0 - (np.mean(similarities) if similarities else 0.0)
                diversity = max(0.0, min(diversity, 1.0))
            else:
                diversity = 0.0
            return embeddings, quality_metadata_list, diversity
        
        # Farthest-point sampling
        vectors = [emb.vector for emb in embeddings]
        n = len(vectors)
        
        selected_indices = [0]
        remaining = set(range(1, n))
        
        while len(selected_indices) < n_select and remaining:
            # Find point farthest from all selected points
            best_dist = -1.0
            best_idx = -1
            
            for i in remaining:
                min_dist = float('inf')
                for j in selected_indices:
                    dist = 1.0 - float(np.dot(vectors[i], vectors[j]))
                    if dist < min_dist:
                        min_dist = dist
                
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_idx = i
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)
            else:
                break
        
        selected = sorted(selected_indices)
        
        selected_embeddings = [embeddings[i] for i in selected]
        selected_metadata = [quality_metadata_list[i] for i in selected]
        
        # Diversity score: mean pairwise cosine distance of selected set
        pairwise_distances = [
            1.0 - float(np.dot(
                selected_embeddings[i].vector,
                selected_embeddings[j].vector
            ))
            for i in range(len(selected_embeddings))
            for j in range(i + 1, len(selected_embeddings))
        ]
        
        mean_distance = np.mean(pairwise_distances) if pairwise_distances else 0.0
        diversity = max(0.0, min(mean_distance, 1.0))
        
        print(f"[RegistrationHandler] Diverse selection: "
              f"{len(embeddings)} → {n_select} frames "
              f"(D={diversity:.4f})")
        
        return selected_embeddings, selected_metadata, diversity
    
    def _capture_frames(self) -> List[FrameData]:
        """
        Capture N × multiplier frames, then select the most diverse subset.

        """
        total_candidates = self.FRAME_COUNT * self.BUFFER_MULTIPLIER
        frames = []
        
        print(f"[RegistrationHandler] Capturing {total_candidates} candidate frames "
              f"over ~{total_candidates * self.FRAME_INTERVAL:.1f}s...")
        
        for i in range(total_candidates):
            frame = self.camera.get_latest_frame()
            
            if frame is not None and frame.verify_integrity():
                frames.append(frame)
                print(f"  Candidate {i+1}/{total_candidates}: captured "
                      f"(#{frame.frame_number})")
            else:
                print(f"  Candidate {i+1}/{total_candidates}: no valid frame")
            
            if i < total_candidates - 1:
                time.sleep(self.FRAME_INTERVAL)
        
        print(f"[RegistrationHandler] Capture complete: "
              f"{len(frames)}/{total_candidates} frames valid")
        
        return frames
    
    # ──────────────────────────────────────────────────
    # INTERNAL: PROCESSING WITH QUALITY
    # ──────────────────────────────────────────────────
    
    def _process_frames_with_quality(
        self, frames: List[FrameData]
    ) -> Tuple[List[EmbeddingData], List[QualityMetadata], List[Dict]]:
        """
        Run detection → liveness → lighting → embedding on each frame.
        
        Returns:
            (embeddings, quality_metadata_list, liveness_results)
        """
        embeddings = []
        quality_metadata_list = []
        liveness_results = []
        
        for i, frame in enumerate(frames):
            # Detect face
            face = self.face_detector.detect_largest_face(frame)
            if face is None:
                print(f"  Frame {i+1}: no face detected")
                liveness_results.append({"is_live": False, "reason": "No face"})
                quality_metadata_list.append(QualityMetadata(
                    frame_index=i,
                    confidence=0.0,
                    liveness_variance=0.0,
                    is_live=False,
                    embedding_quality=0.0,
                    lighting="insufficient",
                    weight=0.0,
                ))
                continue
            
            # Liveness on original frame region
            liveness = self.liveness_gate.check_full_frame_region(
                frame.pixel_array, face.bbox
            )
            
            # Lighting assessment
            lighting = self._assess_lighting(
                liveness.variance, self.liveness_gate.threshold
            )
            
            liveness_results.append({
                "is_live": liveness.is_live,
                "variance": liveness.variance,
                "threshold": liveness.threshold,
                "reason": liveness.reason,
                "lighting": lighting,
            })
            
            if not liveness.is_live:
                print(f"  Frame {i+1}: liveness failed "
                      f"(variance={liveness.variance:.1f}, lighting={lighting})")
                quality_metadata_list.append(QualityMetadata(
                    frame_index=i,
                    confidence=face.confidence,
                    liveness_variance=liveness.variance,
                    is_live=False,
                    embedding_quality=0.0,
                    lighting=lighting,
                    weight=0.0,
                ))
                continue
            
            # Extract embedding
            embedding = self.embedder.extract(face)
            if embedding is None or not embedding.verify_normalization():
                print(f"  Frame {i+1}: embedding extraction failed")
                quality_metadata_list.append(QualityMetadata(
                    frame_index=i,
                    confidence=face.confidence,
                    liveness_variance=liveness.variance,
                    is_live=True,
                    embedding_quality=0.0,
                    lighting=lighting,
                    weight=0.0,
                ))
                continue
            
            embeddings.append(embedding)
            quality_metadata_list.append(QualityMetadata(
                frame_index=i,
                confidence=face.confidence,
                liveness_variance=liveness.variance,
                is_live=True,
                embedding_quality=0.0,  # Computed after centroid
                lighting=lighting,
                weight=0.0,  # Computed after centroid
            ))
            
            print(f"  Frame {i+1}: embedding OK "
                  f"(confidence={face.confidence:.2f}, "
                  f"variance={liveness.variance:.1f}, "
                  f"lighting={lighting})")
        
        return embeddings, quality_metadata_list, liveness_results
    
    # ──────────────────────────────────────────────────
    # INTERNAL: LIGHTING ASSESSMENT
    # ──────────────────────────────────────────────────
    
    def _assess_lighting(self, variance: float, threshold: float) -> str:
        """
        Map liveness variance to lighting condition.

        """
        if threshold <= 0:
            return "unknown"
        
        ratio = variance / threshold
        
        if ratio > self.LIGHTING_EXCELLENT:
            return "excellent"
        elif ratio > self.LIGHTING_GOOD:
            return "good"
        elif ratio > self.LIGHTING_ACCEPTABLE:
            return "acceptable"
        elif ratio > self.LIGHTING_LOW:
            return "low"
        else:
            return "insufficient"
    
    # ──────────────────────────────────────────────────
    # INTERNAL: WEIGHTED CENTROID
    # ──────────────────────────────────────────────────
    
    def _compute_weighted_centroid(
        self,
        embeddings: List[EmbeddingData],
        quality_metadata_list: List[QualityMetadata],
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Compute quality-weighted centroid.
        
        Weight formula:
            w_i = confidence_i × min(variance_i / threshold, 2.0)

        """
        if len(embeddings) == 1:
            return embeddings[0].vector.copy(), [1.0]
        
        threshold = self.liveness_gate.threshold
        
        # Compute weights
        weights = []
        for qm in quality_metadata_list[:len(embeddings)]:
            if qm.is_live and qm.liveness_variance > 0:
                liveness_factor = min(qm.liveness_variance / threshold, 2.0)
                weight = qm.confidence * liveness_factor
            else:
                weight = 0.0
            weights.append(weight)
        
        total_weight = sum(weights)
        
        if total_weight <= 0:
            # All weights zero — fall back to equal weighting
            weights = [1.0] * len(embeddings)
            total_weight = len(embeddings)
        
        # Weighted sum
        summed = np.zeros(512, dtype=np.float32)
        for emb, w in zip(embeddings, weights):
            summed += w * emb.vector
        
        centroid = summed / total_weight
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        # Log
        similarities = [
            float(np.dot(centroid, emb.vector)) for emb in embeddings
        ]
        avg_sim = np.mean(similarities) if similarities else 0
        
        print(f"[RegistrationHandler] Weighted centroid from "
              f"{len(embeddings)} embeddings "
              f"(weights: {[round(w, 2) for w in weights]}, "
              f"avg self-similarity: {avg_sim:.4f})")
        
        return centroid, weights
    
    # ──────────────────────────────────────────────────
    # INTERNAL: QUALITY SCORING
    # ──────────────────────────────────────────────────
    
    def _compute_enrollment_quality(
        self,
        quality_metadata_list: List[QualityMetadata],
        centroid_vector: np.ndarray,
        embeddings: List[EmbeddingData],
        pose_diversity: Optional[float] = None,
    ) -> EnrollmentQuality:
        """
        Compute composite enrollment quality score.
        
        Formula:
            Q = 0.30 × mean_confidence
              + 0.30 × lighting_score
              + 0.20 × pose_diversity
              + 0.20 × mean_self_similarity
        
        Args:
            pose_diversity: Pre-computed diversity score (0.0–1.0).
                            If None, computed from embedding similarities.
        """
        live_metadata = [qm for qm in quality_metadata_list if qm.is_live]
        
        if not live_metadata:
            return EnrollmentQuality(
                overall_score=0.0,
                grade="poor",
                avg_confidence=0.0,
                avg_liveness_variance=0.0,
                lighting_condition="insufficient",
                pose_diversity=0.0,
                avg_self_similarity=0.0,
                frames_captured=len(quality_metadata_list),
                frames_used=0,
                recommendation="No valid frames. Ensure face is visible "
                               "and lighting is adequate."
            )
        
        # Component scores
        avg_confidence = np.mean([qm.confidence for qm in live_metadata])
        avg_variance = np.mean([qm.liveness_variance for qm in live_metadata])
        
        # Lighting score: map variance ratio to 0–1
        threshold = self.liveness_gate.threshold
        lighting_ratio = avg_variance / threshold if threshold > 0 else 0
        lighting_score = min(lighting_ratio / 2.0, 1.0)  # Cap at 1.0
        
        # Overall lighting condition
        lighting_condition = self._assess_lighting(avg_variance, threshold)
        
        # Pose diversity: use pre-computed value or compute from embeddings
        if pose_diversity is not None:
            diversity = pose_diversity
        elif len(embeddings) >= 2:
            similarities = [
                float(np.dot(centroid_vector, emb.vector))
                for emb in embeddings
            ]
            diversity = 1.0 - np.mean(similarities)  # Lower similarity = more diverse
            diversity = max(0.0, min(diversity, 1.0))
        else:
            diversity = 0.0
        
        # Self-similarity
        if embeddings:
            self_sims = [
                float(np.dot(centroid_vector, emb.vector))
                for emb in embeddings
            ]
            avg_self_similarity = np.mean(self_sims)
        else:
            avg_self_similarity = 0.0
        
        # Composite score
        overall = (
            0.30 * avg_confidence +
            0.30 * lighting_score +
            0.20 * diversity +
            0.20 * avg_self_similarity
        )
        overall = max(0.0, min(overall, 1.0))
        
        # Grade
        if overall >= self.QUALITY_GRADE_EXCELLENT:
            grade = "excellent"
            recommendation = "Identity is strong. Ready for identification."
        elif overall >= self.QUALITY_GRADE_GOOD:
            grade = "good"
            recommendation = "Good enrollment. Should work reliably."
        elif overall >= self.QUALITY_GRADE_MARGINAL:
            grade = "marginal"
            recommendation = (
                "Enrollment is acceptable but could be improved. "
                "Consider re-enrolling with better lighting."
            )
        else:
            grade = "poor"
            # Specific recommendations based on weak component
            if lighting_score < 0.4:
                recommendation = (
                    "Poor lighting. Move to a brighter area or face a window."
                )
            elif avg_confidence < 0.7:
                recommendation = (
                    "Low detection confidence. Face the camera directly."
                )
            elif avg_self_similarity < 0.85:
                recommendation = (
                    "Inconsistent embeddings. Stay still during capture."
                )
            else:
                recommendation = (
                    "Enrollment quality is low. Consider re-enrolling."
                )
        
        return EnrollmentQuality(
            overall_score=round(float(overall), 4),
            grade=grade,
            avg_confidence=round(float(avg_confidence), 4),
            avg_liveness_variance=round(float(avg_variance), 2),
            lighting_condition=lighting_condition,
            pose_diversity=round(float(diversity), 4),
            avg_self_similarity=round(float(avg_self_similarity), 4),
            frames_captured=len(quality_metadata_list),
            frames_used=len(embeddings),
            recommendation=recommendation,
        )
    
    # ──────────────────────────────────────────────────
    # INTERNAL: LAYERED DUPLICATE CHECK
    # ──────────────────────────────────────────────────
    
    def _check_duplicate_layered(
        self,
        centroid_embedding: EmbeddingData,
        embeddings: List[EmbeddingData],
        quality_metadata_list: List[QualityMetadata],
        quality: EnrollmentQuality,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Three-layer duplicate detection.
        """
        if self.database.count() == 0:
            return False, None
        
        # Layer 1: Centroid match
        centroid_matches = self.database.query_nearest(
            centroid_embedding, n_results=1
        )
        if centroid_matches:
            match = centroid_matches[0]
            if match['distance'] < self.DUPLICATE_THRESHOLD_CENTROID:
                return True, match
        
        # Layer 2: Best individual frame (highest-weighted frame)
        best_embedding, best_weight = self._find_best_embedding(
            embeddings, quality_metadata_list
        )
        if best_embedding:
            frame_matches = self.database.query_nearest(
                best_embedding, n_results=1
            )
            if frame_matches:
                match = frame_matches[0]
                if match['distance'] < self.DUPLICATE_THRESHOLD_BEST_FRAME:
                    return True, match
        
        # Layer 3: Quality-adjusted threshold
        if quality.overall_score < self.QUALITY_GRADE_MARGINAL:
            if centroid_matches:
                match = centroid_matches[0]
                if match['distance'] < self.DUPLICATE_THRESHOLD_LOW_QUALITY:
                    return True, match
        
        return False, None
    
    def _find_best_embedding(
        self,
        embeddings: List[EmbeddingData],
        quality_metadata_list: List[QualityMetadata],
    ) -> Tuple[Optional[EmbeddingData], float]:
        """Find the highest-weighted frame from the batch."""
        if not embeddings or not quality_metadata_list:
            return None, 0.0
        
        best = None
        best_weight = -1.0
        
        for i, emb in enumerate(embeddings):
            if i >= len(quality_metadata_list):
                break
            weight = quality_metadata_list[i].weight
            if weight > best_weight:
                best_weight = weight
                best = emb
        
        return best, best_weight
    
    # ──────────────────────────────────────────────────
    # INTERNAL: HELPERS
    # ──────────────────────────────────────────────────
    
    def _summarize_liveness(self, liveness_results: List[Dict]) -> Dict:
        """Create a summary liveness dict from multiple frame results."""
        if not liveness_results:
            return {"is_live": False, "reason": "No liveness data"}
        
        live_count = sum(1 for l in liveness_results if l.get("is_live"))
        variances = [
            l.get("variance", 0) for l in liveness_results
            if l.get("variance")
        ]
        
        return {
            "is_live": live_count > 0,
            "frames_checked": len(liveness_results),
            "frames_live": live_count,
            "avg_variance": round(float(np.mean(variances)), 2) if variances else 0,
            "threshold": liveness_results[0].get("threshold", 0) if liveness_results else 0,
        }
    
    def _fail(
        self,
        user_id: str,
        name: str,
        start_time: float,
        error: str,
        liveness: Optional[Dict] = None,
        quality: Optional[EnrollmentQuality] = None,
    ) -> RegistrationResult:
        """Build a failed RegistrationResult consistently."""
        processing_time = (time.time() - start_time) * 1000
        
        kwargs = {
            "status": "failed",
            "user_id": user_id,
            "name": name,
            "error": error,
            "processing_time_ms": round(processing_time, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        if liveness:
            kwargs["liveness"] = liveness
        
        if quality:
            kwargs["quality"] = {
                "overall_score": quality.overall_score,
                "grade": quality.grade,
                "lighting_condition": quality.lighting_condition,
                "frames_used": quality.frames_used,
                "frames_captured": quality.frames_captured,
                "recommendation": quality.recommendation,
            }
            kwargs["quality_score"] = quality.overall_score
            kwargs["quality_grade"] = quality.grade
            kwargs["lighting_condition"] = quality.lighting_condition
            kwargs["frames_used"] = quality.frames_used
            kwargs["frames_captured"] = quality.frames_captured
            kwargs["recommendation"] = quality.recommendation
        
        return RegistrationResult(**kwargs)
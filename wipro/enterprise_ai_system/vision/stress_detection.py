"""
Computer Vision Module for Stress Detection
Analyze facial features and eye focus to detect fatigue and stress indicators
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FatigueDetector:
    """Detect fatigue and stress indicators using computer vision"""
    
    def __init__(self):
        """Initialize MediaPipe Face Detection and Face Mesh"""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks indices
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    def detect_face(self, frame: np.ndarray) -> bool:
        """
        Detect if face is present in frame
        
        Args:
            frame: Input video frame (BGR)
        
        Returns:
            Boolean indicating face presence
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        return results.detections is not None and len(results.detections) > 0
    
    def calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection
        
        Args:
            eye_landmarks: Array of eye landmark coordinates
        
        Returns:
            Eye aspect ratio
        """
        if len(eye_landmarks) < 6:
            return 1.0
        
        # Calculate distances between landmarks
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR = (A + B) / (2 * C)
        ear = (A + B) / (2.0 * C + 1e-6)
        return ear
    
    def detect_blinks(self, frame: np.ndarray, 
                     blink_threshold: float = 0.2) -> Tuple[int, float]:
        """
        Detect eye blinks in frame
        
        Args:
            frame: Input video frame
            blink_threshold: EAR threshold for blink detection
        
        Returns:
            Tuple of (blink_count, average_ear)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        blink_count = 0
        avg_ear = 1.0
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Get eye landmarks
            left_eye = np.array([
                [landmarks[i].x, landmarks[i].y] 
                for i in self.LEFT_EYE_INDICES
            ])
            right_eye = np.array([
                [landmarks[i].x, landmarks[i].y] 
                for i in self.RIGHT_EYE_INDICES
            ])
            
            # Calculate EAR
            left_ear = self.calculate_eye_aspect_ratio(left_eye)
            right_ear = self.calculate_eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Count blinks
            if avg_ear < blink_threshold:
                blink_count = 1
        
        return blink_count, avg_ear
    
    def detect_head_pose(self, frame: np.ndarray) -> Dict:
        """
        Estimate head pose (pitch, yaw, roll)
        
        Args:
            frame: Input video frame
        
        Returns:
            Dictionary with head pose angles
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        pose = {
            'pitch': 0.0,
            'yaw': 0.0,
            'roll': 0.0,
            'detected': False
        }
        
        if results.multi_face_landmarks:
            landmarks = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.multi_face_landmarks[0].landmark
            ])
            
            # Key points for head pose estimation
            # 33 and 263 are eye outer corners
            # 1 is nose tip
            # 152 and 376 are mouth corners
            
            # Simple estimation: yaw based on eye to nose distance
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            nose = landmarks[1]
            
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            yaw = (nose[0] - eye_center_x) * 100  # Scale for visibility
            
            # Pitch based on nose to eyes vertical distance
            pitch = (nose[1] - (left_eye[1] + right_eye[1]) / 2) * 100
            
            pose['yaw'] = float(np.clip(yaw, -30, 30))
            pose['pitch'] = float(np.clip(pitch, -30, 30))
            pose['detected'] = True
        
        return pose
    
    def calculate_focus_score(self, frame: np.ndarray) -> float:
        """
        Calculate overall focus score (0-100)
        Based on eye openness and head posture
        
        Args:
            frame: Input video frame
        
        Returns:
            Focus score (0-100)
        """
        # Detect blinks
        _, avg_ear = self.detect_blinks(frame)
        
        # Detect head pose
        pose = self.detect_head_pose(frame)
        
        # Calculate focus score
        # Eyes open (EAR > 0.2) = good focus
        eye_score = min(avg_ear / 0.3, 1.0) * 100
        
        # Good head posture (near neutral) = good focus
        if pose['detected']:
            head_angle = np.sqrt(pose['yaw']**2 + pose['pitch']**2)
            head_score = max(100 - head_angle * 2, 0)
        else:
            head_score = 50
        
        # Combined score
        focus_score = (eye_score * 0.6 + head_score * 0.4)
        
        return float(np.clip(focus_score, 0, 100))
    
    def detect_stress_indicators(self, frame: np.ndarray) -> Dict:
        """
        Comprehensive stress indicator detection
        
        Args:
            frame: Input video frame
        
        Returns:
            Dictionary with stress indicators
        """
        indicators = {
            'face_present': self.detect_face(frame),
            'focus_score': 0.0,
            'head_pose': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0, 'detected': False},
            'blink_count': 0,
            'eye_aspect_ratio': 0.0,
            'stress_level': 'normal'
        }
        
        if indicators['face_present']:
            # Calculate focus score
            indicators['focus_score'] = self.calculate_focus_score(frame)
            
            # Detect head pose
            indicators['head_pose'] = self.detect_head_pose(frame)
            
            # Detect blinks
            blinks, ear = self.detect_blinks(frame)
            indicators['blink_count'] = blinks
            indicators['eye_aspect_ratio'] = ear
            
            # Determine stress level
            if indicators['focus_score'] < 30:
                indicators['stress_level'] = 'critical'
            elif indicators['focus_score'] < 50:
                indicators['stress_level'] = 'high'
            elif indicators['focus_score'] < 70:
                indicators['stress_level'] = 'moderate'
            else:
                indicators['stress_level'] = 'normal'
        
        return indicators
    
    def process_video_stream(self, video_source: int = 0, 
                            duration: int = 30) -> Dict:
        """
        Process video stream from webcam and analyze stress indicators
        
        Args:
            video_source: Video source (0 for default webcam)
            duration: Duration to capture in seconds
        
        Returns:
            Dictionary with aggregated stress indicators
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error("Cannot open video source")
            return {'error': 'Cannot open video source'}
        
        # Get FPS and calculate frame count
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = fps * duration
        
        focus_scores = []
        stress_levels = {'normal': 0, 'moderate': 0, 'high': 0, 'critical': 0}
        frames_processed = 0
        
        try:
            while frames_processed < frame_count:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Resize frame for faster processing
                frame = cv2.resize(frame, (640, 480))
                
                # Detect stress indicators
                indicators = self.detect_stress_indicators(frame)
                
                if indicators['face_present']:
                    focus_scores.append(indicators['focus_score'])
                    stress_levels[indicators['stress_level']] += 1
                
                frames_processed += 1
                
                # Display frame with indicators
                display_frame = self._draw_indicators(frame, indicators)
                cv2.imshow('Stress Detection', display_frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Aggregate results
        result = {
            'frames_processed': frames_processed,
            'average_focus_score': np.mean(focus_scores) if focus_scores else 0,
            'min_focus_score': np.min(focus_scores) if focus_scores else 0,
            'max_focus_score': np.max(focus_scores) if focus_scores else 0,
            'stress_level_distribution': stress_levels,
            'face_detection_rate': len(focus_scores) / max(frames_processed, 1)
        }
        
        return result
    
    @staticmethod
    def _draw_indicators(frame: np.ndarray, indicators: Dict) -> np.ndarray:
        """Draw stress indicators on frame"""
        display_frame = frame.copy()
        
        if indicators['face_present']:
            # Draw focus score
            text = f"Focus: {indicators['focus_score']:.1f}%"
            cv2.putText(display_frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw stress level
            stress_color = {
                'normal': (0, 255, 0),
                'moderate': (255, 165, 0),
                'high': (255, 100, 0),
                'critical': (0, 0, 255)
            }
            color = stress_color.get(indicators['stress_level'], (255, 255, 255))
            text = f"Stress: {indicators['stress_level'].upper()}"
            cv2.putText(display_frame, text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw head pose
            pose = indicators['head_pose']
            if pose['detected']:
                text = f"Yaw: {pose['yaw']:.1f}° Pitch: {pose['pitch']:.1f}°"
                cv2.putText(display_frame, text, (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(display_frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return display_frame

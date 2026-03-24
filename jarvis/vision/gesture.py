"""Hand gesture recognition module for JARVIS.

Uses MediaPipe for real-time hand tracking and gesture classification.
"""

import logging
import threading
import time
from typing import Any

logger = logging.getLogger("jarvis.vision.gesture")

# Gesture name constants
GESTURE_OPEN_PALM = "open_palm"
GESTURE_CLOSED_FIST = "closed_fist"
GESTURE_THUMBS_UP = "thumbs_up"
GESTURE_THUMBS_DOWN = "thumbs_down"
GESTURE_PEACE_SIGN = "peace_sign"
GESTURE_POINT_UP = "point_up"
GESTURE_POINT_DOWN = "point_down"
GESTURE_NONE = "none"


class GestureRecognizer:
    """Real-time hand gesture recognition using MediaPipe."""

    def __init__(
        self,
        camera_index: int = 0,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        """Initialize the gesture recognizer.

        Args:
            camera_index: Camera device index.
            min_detection_confidence: Minimum hand detection confidence.
            min_tracking_confidence: Minimum hand tracking confidence.
        """
        self.camera_index = camera_index
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._running = False
        self._current_gesture: str | None = None
        self._lock = threading.Lock()
        self._hands: Any = None
        self._cap: Any = None
        self._initialized = False
        self._last_gesture_time = 0.0
        self._gesture_cooldown = 1.0  # Seconds between gesture detections

    def initialize(self) -> bool:
        """Initialize MediaPipe and camera.

        Returns:
            True if initialization succeeded.
        """
        try:
            import cv2
            import mediapipe as mp

            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

            # Test camera access
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                logger.warning(
                    f"Camera {self.camera_index} not available. "
                    "Gesture recognition disabled."
                )
                self._cap = None
                return False

            self._initialized = True
            logger.info("Gesture recognition initialized")
            return True

        except ImportError as e:
            logger.warning(f"MediaPipe/OpenCV not available: {e}. Gesture recognition disabled.")
            return False
        except Exception as e:
            logger.error(f"Gesture initialization failed: {e}")
            return False

    def detect(self) -> str | None:
        """Detect a hand gesture from camera input.

        Returns:
            Gesture name string or None if no gesture detected.
        """
        if not self._initialized:
            if not self.initialize():
                time.sleep(1)
                return None

        # Cooldown check
        current_time = time.time()
        if current_time - self._last_gesture_time < self._gesture_cooldown:
            time.sleep(0.1)
            return None

        try:
            import cv2

            ret, frame = self._cap.read()
            if not ret:
                return None

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                gesture = self._classify_gesture(hand_landmarks)

                if gesture != GESTURE_NONE:
                    self._last_gesture_time = current_time
                    with self._lock:
                        self._current_gesture = gesture
                    logger.info(f"Gesture detected: {gesture}")
                    return gesture

        except Exception as e:
            logger.error(f"Gesture detection error: {e}")
            time.sleep(0.5)

        return None

    def _classify_gesture(self, hand_landmarks: Any) -> str:
        """Classify hand gesture from landmarks.

        Uses finger tip positions relative to finger joints to determine
        which fingers are extended, then maps to gesture names.

        Args:
            hand_landmarks: MediaPipe hand landmarks.

        Returns:
            Gesture name string.
        """
        landmarks = hand_landmarks.landmark

        # Finger tip and pip (proximal interphalangeal) landmark indices
        tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        pip_ids = [3, 6, 10, 14, 18]

        fingers_up = []

        # Thumb: compare x position (left/right)
        if landmarks[tip_ids[0]].x < landmarks[pip_ids[0]].x:
            fingers_up.append(True)
        else:
            fingers_up.append(False)

        # Other fingers: compare y position (up/down)
        for i in range(1, 5):
            if landmarks[tip_ids[i]].y < landmarks[pip_ids[i]].y:
                fingers_up.append(True)
            else:
                fingers_up.append(False)

        count_up = sum(fingers_up)

        # Classify gestures
        if count_up == 5:
            return GESTURE_OPEN_PALM
        elif count_up == 0:
            return GESTURE_CLOSED_FIST
        elif fingers_up[0] and count_up == 1:
            # Check if thumb is up or down
            if landmarks[tip_ids[0]].y < landmarks[pip_ids[0]].y:
                return GESTURE_THUMBS_UP
            else:
                return GESTURE_THUMBS_DOWN
        elif fingers_up[1] and fingers_up[2] and count_up == 2:
            return GESTURE_PEACE_SIGN
        elif fingers_up[1] and count_up == 1:
            # Index finger pointing
            if landmarks[tip_ids[1]].y < landmarks[0].y:
                return GESTURE_POINT_UP
            else:
                return GESTURE_POINT_DOWN

        return GESTURE_NONE

    def start(self, callback: Any = None) -> None:
        """Start continuous gesture detection in background.

        Args:
            callback: Function to call with detected gesture names.
        """
        self._running = True
        self._detect_thread = threading.Thread(
            target=self._detection_loop,
            args=(callback,),
            daemon=True,
        )
        self._detect_thread.start()
        logger.info("Gesture detection started")

    def _detection_loop(self, callback: Any = None) -> None:
        """Background loop for continuous gesture detection.

        Args:
            callback: Callback for detected gestures.
        """
        while self._running:
            gesture = self.detect()
            if gesture and callback:
                callback(gesture)
            time.sleep(0.05)  # ~20 FPS

    def stop(self) -> None:
        """Stop gesture detection and release camera."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
        if self._hands is not None:
            self._hands.close()
        self._initialized = False
        logger.info("Gesture recognition stopped")

    @property
    def current_gesture(self) -> str | None:
        """Get the most recently detected gesture."""
        with self._lock:
            return self._current_gesture

    @property
    def is_initialized(self) -> bool:
        """Check if gesture recognition is initialized."""
        return self._initialized

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = Path(__file__).resolve().with_name("pose_landmarker_lite.task")

def create_detector():
    if not MODEL_PATH.exists():
        print(f"Downloading pose model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def angle_between(a, b, c, frame_size=(1, 1), world_coordinates=False):
    # Normalized x and y have different scales on a rectangular camera frame.
    width, height = frame_size
    if world_coordinates:
        # World x/y/z share a metric scale; include depth to avoid foreshortening.
        coordinates = np.array([[p.x, p.y, p.z] for p in (a, b, c)])
        if not np.all(np.isfinite(coordinates)):
            return None
        ba = coordinates[0] - coordinates[1]
        bc = coordinates[2] - coordinates[1]
    else:
        ba = np.array([(a.x - b.x) * width, (a.y - b.y) * height])
        bc = np.array([(c.x - b.x) * width, (c.y - b.y) * height])

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if not np.isfinite(denominator) or denominator < 1e-8:
        return None
    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def knee_angles(landmarks, frame_size=(1, 1), world_landmarks=None):
    """Measure only legs whose hip, knee and ankle are reliably visible."""
    if len(landmarks) < 29:
        return []
    if world_landmarks is not None and len(world_landmarks) < 29:
        return []
    angles = []
    for indices in ((23, 25, 27), (24, 26, 28)):
        leg = [landmarks[index] for index in indices]
        if not all(
            getattr(point, "visibility", 1.0) >= 0.5
            and getattr(point, "presence", 1.0) >= 0.5
            and 0 <= point.x <= 1 and 0 <= point.y <= 1
            for point in leg
        ):
            continue
        if world_landmarks is not None:
            leg = [world_landmarks[index] for index in indices]
            if not all(
                getattr(point, "visibility", 1.0) >= 0.5
                and getattr(point, "presence", 1.0) >= 0.5
                for point in leg
            ):
                continue
        angle = angle_between(*leg, frame_size=frame_size,
                              world_coordinates=world_landmarks is not None)
        if angle is not None:
            angles.append(angle)
    return angles


def posture_status(landmarks, frame_size=(1, 1), world_landmarks=None):
    angles = knee_angles(landmarks, frame_size, world_landmarks)
    if not angles:
        return "Tracking lost"
    # Use separate thresholds so a partial bend cannot rearm the counter.
    if max(angles) <= 110:
        return "Squatting"
    if min(angles) >= 160:
        return "Standing"
    return "Moving"


def squat_state(landmarks, frame_size=(1, 1), world_landmarks=None):
    return posture_status(landmarks, frame_size, world_landmarks) == "Squatting"


class SquatCounter:
    """Count confirmed standing -> squat -> standing cycles."""

    def __init__(self):
        self.count = 0
        self.status = "Tracking lost"
        self._phase = "start"
        self._candidate = None
        self._frames = 0

    def update(self, status):
        if status == "Tracking lost":
            # An unseen movement cannot complete a verifiable rep.
            self._phase = "start"
            self._candidate = None
            self._frames = 0
            self.status = status
            return
        if status != self._candidate:
            self._candidate = status
            self._frames = 0
        self._frames = min(self._frames + 1, 4)
        self.status = status if self._frames >= 4 else "Moving"
        if self._frames < 4:
            return
        if status == "Standing":
            if self._phase == "down":
                self.count += 1
            self._phase = "ready"
        elif status == "Squatting" and self._phase == "ready":
            self._phase = "down"


def draw_pose(frame, landmarks):
    if len(landmarks) < 2:
        return

    height, width, _ = frame.shape
    pose_connections = [
        (0, 1), (0, 4), (1, 2), (2, 3), (3, 7),
        (4, 5), (5, 6), (6, 8), (8, 10), (5, 9),
        (11, 12), (11, 23), (12, 24), (23, 24),
        (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32),
        (11, 13), (13, 15), (15, 17), (12, 14), (14, 16), (16, 18),
        (17, 19), (18, 20), (11, 12), (13, 14)
    ]

    for start_idx, end_idx in pose_connections:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        x1 = int(start.x * width)
        y1 = int(start.y * height)
        x2 = int(end.x * width)
        y2 = int(end.y * height)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def main():
    detector = create_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        detector.close()
        cap.release()
        raise RuntimeError("Could not open webcam")

    counter = SquatCounter()
    last_timestamp_ms = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(last_timestamp_ms + 1, int(time.monotonic() * 1000))
        last_timestamp_ms = timestamp_ms
        result = detector.detect_for_video(mp_image, timestamp_ms)

        status = "Tracking lost"
        angles = []
        angle_mode = "2D"
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            frame_size = (frame.shape[1], frame.shape[0])
            world_landmarks = (
                result.pose_world_landmarks[0] if result.pose_world_landmarks else None
            )
            angle_mode = "3D" if world_landmarks is not None else "2D"
            status = posture_status(landmarks, frame_size, world_landmarks)
            angles = knee_angles(landmarks, frame_size, world_landmarks)
            draw_pose(frame, landmarks)

        counter.update(status)
        cv2.putText(
            frame,
            f"Squats: {counter.count}  |  {counter.status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        feedback = (
            f"{angle_mode} knee angles: " + " / ".join(f"{angle:.0f}" for angle in angles)
            if angles else "Keep hips, knees and ankles in view"
        )
        cv2.putText(frame, feedback, (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        cv2.imshow("Squat Counter", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

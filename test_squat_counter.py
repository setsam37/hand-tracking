import math
from types import SimpleNamespace
import unittest

import numpy as np
import squat_counter


namespace = vars(squat_counter)


def pose(angle, visibility=1.0):
    landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=visibility, presence=1.0)
                 for _ in range(33)]
    for hip, knee, ankle, x in ((23, 25, 27, 0.4), (24, 26, 28, 0.6)):
        landmarks[knee].x, landmarks[knee].y = x, 0.6
        landmarks[ankle].x, landmarks[ankle].y = x, 0.8
        landmarks[hip].x = x + 0.2 * math.sin(math.radians(angle))
        landmarks[hip].y = 0.6 + 0.2 * math.cos(math.radians(angle))
    return landmarks


def rotated_pose(angle, yaw):
    """Project the same 3D leg bend at different camera-facing rotations."""
    world = pose(angle)
    image = pose(angle)
    for hip, knee, ankle, x in ((23, 25, 27, -0.1), (24, 26, 28, 0.1)):
        for index, y, z in (
            (hip, 0.2 * math.cos(math.radians(angle)),
             0.2 * math.sin(math.radians(angle))),
            (knee, 0.0, 0.0), (ankle, 0.2, 0.0),
        ):
            rotation = math.radians(yaw)
            point = world[index]
            point.x = x * math.cos(rotation) + z * math.sin(rotation)
            point.y = y
            point.z = -x * math.sin(rotation) + z * math.cos(rotation)
            image[index].x = 0.5 + point.x
            image[index].y = 0.6 + point.y
    return image, world


class PostureTests(unittest.TestCase):
    def test_3d_squat_is_detected_from_front_diagonal_and_side(self):
        for yaw in (0, 45, 90, 180):
            with self.subTest(yaw=yaw):
                image, world = rotated_pose(100, yaw)
                self.assertEqual(squat_counter.posture_status(
                    image, world_landmarks=world), "Squatting")

    def test_world_coordinates_do_not_bypass_image_visibility_checks(self):
        image, world = rotated_pose(100, 0)
        for index in (23, 24):
            image[index].visibility = 0.1
        self.assertEqual(squat_counter.posture_status(
            image, world_landmarks=world), "Tracking lost")

    def test_invalid_world_coordinates_do_not_classify_posture(self):
        for invalid in (float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                image, world = rotated_pose(100, 0)
                for index in (25, 26):
                    world[index].z = invalid
                self.assertEqual(squat_counter.posture_status(
                    image, world_landmarks=world), "Tracking lost")

    def test_squat_does_not_require_hips_below_knees(self):
        self.assertTrue(namespace["squat_state"](pose(100)))

    def test_straight_legs_are_not_a_squat(self):
        self.assertFalse(namespace["squat_state"](pose(175)))

    def test_unreliable_landmarks_are_not_a_squat(self):
        self.assertFalse(namespace["squat_state"](pose(80, visibility=0.1)))

    def test_zero_length_leg_is_not_an_angle(self):
        point = SimpleNamespace(x=0.5, y=0.5)
        with np.errstate(all="ignore"):
            self.assertIsNone(namespace["angle_between"](point, point, point))

    def test_missing_pose_is_not_standing(self):
        self.assertIn("posture_status", tuple(namespace))
        self.assertEqual(namespace["posture_status"]([]), "Tracking lost")

    def test_partial_bend_is_not_standing(self):
        self.assertIn("posture_status", tuple(namespace))
        self.assertEqual(namespace["posture_status"](pose(135)), "Moving")

    def test_angles_account_for_frame_aspect_ratio(self):
        # In a 2:1 frame, these vectors are (1, -1) and (0, 1): 135 degrees.
        a = SimpleNamespace(x=0.75, y=0.0)
        b = SimpleNamespace(x=0.5, y=0.5)
        c = SimpleNamespace(x=0.5, y=1.0)
        import inspect
        self.assertIn("frame_size", inspect.signature(namespace["angle_between"]).parameters)
        self.assertAlmostEqual(namespace["angle_between"](a, b, c, (2, 1)), 135)


class CounterTests(unittest.TestCase):
    def test_front_facing_squat_cycle_counts_one_rep(self):
        tracker = self.tracker()
        for angle in (175, 135, 100, 135, 175):
            image, world = rotated_pose(angle, 0)
            self.feed(tracker, squat_counter.posture_status(
                image, world_landmarks=world))
        self.assertEqual(tracker.count, 1)

    def tracker(self):
        self.assertIn("SquatCounter", tuple(namespace), "Counting needs a bounded cycle tracker")
        return namespace["SquatCounter"]()

    def feed(self, tracker, status, frames=5):
        for _ in range(frames):
            tracker.update(status)

    def test_counts_only_after_return_to_standing(self):
        tracker = self.tracker()
        self.feed(tracker, "Standing")
        self.feed(tracker, "Squatting")
        self.assertEqual(tracker.count, 0)
        self.feed(tracker, "Moving")
        self.feed(tracker, "Standing")
        self.assertEqual(tracker.count, 1)

    def test_long_hold_does_not_delay_next_rep(self):
        tracker = self.tracker()
        for _ in range(2):
            self.feed(tracker, "Standing")
            self.feed(tracker, "Squatting", 300)
            self.feed(tracker, "Standing")
        self.assertEqual(tracker.count, 2)

    def test_lost_tracking_cancels_incomplete_rep(self):
        tracker = self.tracker()
        self.feed(tracker, "Standing")
        self.feed(tracker, "Squatting")
        self.feed(tracker, "Tracking lost")
        self.feed(tracker, "Standing")
        self.assertEqual(tracker.count, 0)

    def test_one_frame_noise_does_not_count(self):
        tracker = self.tracker()
        self.feed(tracker, "Standing")
        self.feed(tracker, "Squatting", 1)
        self.feed(tracker, "Standing")
        self.assertEqual(tracker.count, 0)

    def test_starting_in_squat_does_not_count(self):
        tracker = self.tracker()
        self.feed(tracker, "Squatting")
        self.feed(tracker, "Standing")
        self.assertEqual(tracker.count, 0)


if __name__ == "__main__":
    unittest.main()

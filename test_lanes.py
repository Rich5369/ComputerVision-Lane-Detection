import sys
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import lanes


LEFT = np.array([0.0, -0.5, 0.36 + 0.5 * 0.72])  # x = 0.36 at the bottom, leaning right going up.
RIGHT = np.array([0.0, 0.5, 0.62 - 0.5 * 0.72])  # x = 0.62 at the bottom; they converge upward.


def confirm(tracker, **models):
    """Feed enough consistent detections for the tracker to start showing lanes."""
    for _ in range(lanes.CONFIRM_FRAMES):
        tracker.update({side: (model, lanes.SEED_TOP, 1) for side, model in models.items()})


class LaneDetectionTests(unittest.TestCase):
    def test_dashed_gap_is_held_briefly_then_expires(self):
        tracker = lanes.LaneTracker()
        model = np.array([0.0, -0.5, 0.75])
        confirm(tracker, left=model)
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)

        for _ in range(lanes.MAX_MISSING_FRAMES):
            tracker.update({})
            self.assertIn("left", tracker.models)
            self.assertGreater(np.count_nonzero(lanes.draw_lanes(blank, tracker)), 0)

        tracker.update({})
        self.assertNotIn("left", tracker.models)
        self.assertEqual(np.count_nonzero(lanes.draw_lanes(blank, tracker)), 0)

    def test_small_changes_are_smoothed_and_large_jump_is_rejected(self):
        tracker = lanes.LaneTracker()
        original = np.array([0.0, -0.5, 0.75])
        confirm(tracker, left=original)
        shifted = np.array([0.0, -0.5, 0.79])
        tracker.update({"left": (shifted, lanes.SEED_TOP, 1)})
        self.assertGreater(tracker.models["left"][2], original[2])
        self.assertLess(tracker.models["left"][2], shifted[2])

        before_jump = tracker.models["left"].copy()
        tracker.update({"left": (np.array([0.0, -0.5, 1.1]), lanes.SEED_TOP, 1)})
        np.testing.assert_array_equal(tracker.models["left"], before_jump)
        self.assertEqual(tracker.missing["left"], 1)

    def test_dashed_curve_follows_markings(self):
        height, width = 720, 1280
        mask = np.zeros((height, width), dtype=np.uint8)

        def lane_x(y_fraction):
            return 0.48 - 0.48 * (y_fraction - 0.42) + 0.9 * (y_fraction - 0.57) ** 2

        for start in range(int(height * lanes.ROI_TOP), int(height * lanes.DRAW_BOTTOM), 38):
            points = []
            for row in range(start, min(start + 24, int(height * lanes.DRAW_BOTTOM))):
                points.append((round(lane_x(row / height) * width), row))
            cv2.polylines(mask, [np.array(points, dtype=np.int32)], False, 255, 8)

        guide = np.array([0.0, -0.48, 0.69])
        result = lanes.fit_lane_curve(mask, "left", guide)
        self.assertIsNotNone(result)
        model, top, confidence = result
        self.assertLess(top, 0.56)
        self.assertGreater(confidence, lanes.MIN_FIT_CONFIDENCE)
        self.assertGreater(abs(model[0]), 0.3)
        for sample_y in (0.45, 0.57, 0.69):
            self.assertLess(abs(np.polyval(model, sample_y) - lane_x(sample_y)), 0.025)

    def test_roi_reaches_farther_up_without_sky(self):
        image = np.full((720, 1280), 255, dtype=np.uint8)
        masked = lanes.region_of_interest(image)
        self.assertEqual(masked[int(720 * 0.52), 640], 255)
        self.assertEqual(masked[int(720 * 0.45), 640], 0)
        self.assertEqual(masked[int(720 * 0.8), 640], 0)

    def test_processing_preserves_original_frame_size(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = lanes.process_frame(frame)
        self.assertEqual(result.shape, frame.shape)

    def test_q_stops_playback_and_releases_video(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 30
        capture.read.side_effect = [(True, frame), (True, frame)]

        with (patch.object(sys, "argv", ["lanes.py"]),
              patch.object(lanes.cv2, "VideoCapture", return_value=capture),
              patch.object(lanes.cv2, "namedWindow"),
              patch.object(lanes.cv2, "resizeWindow"),
              patch.object(lanes.cv2, "imshow"),
              patch.object(lanes.cv2, "waitKey", return_value=ord("q")),
              patch.object(lanes.cv2, "destroyAllWindows") as close,
              patch.object(lanes, "process_frame", return_value=frame)):
            lanes.main()

        self.assertEqual(capture.read.call_count, 1)
        capture.release.assert_called_once()
        close.assert_called_once()

    def test_debug_flag_opens_a_second_window_only_when_requested(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for argv, windows in ((["lanes.py"], 1), (["lanes.py", "--debug"], 2)):
            capture = Mock()
            capture.isOpened.return_value = True
            capture.get.return_value = 30
            capture.read.side_effect = [(True, frame), (True, frame)]
            with (patch.object(sys, "argv", argv),
                  patch.object(lanes.cv2, "VideoCapture", return_value=capture),
                  patch.object(lanes.cv2, "namedWindow"),
                  patch.object(lanes.cv2, "resizeWindow"),
                  patch.object(lanes.cv2, "imshow") as show,
                  patch.object(lanes.cv2, "waitKey", return_value=ord("q")),
                  patch.object(lanes.cv2, "destroyAllWindows")):
                lanes.main()
            self.assertEqual(show.call_count, windows)

    def test_new_lane_needs_consecutive_confirmation(self):
        tracker = lanes.LaneTracker()
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        tracker.update({"left": (LEFT, lanes.SEED_TOP, 1)})
        self.assertNotIn("left", tracker.models)
        self.assertEqual(np.count_nonzero(lanes.draw_lanes(blank, tracker)), 0)

        tracker.update({})  # A miss restarts the count, so a one-off ghost never appears.
        for _ in range(lanes.CONFIRM_FRAMES - 1):
            tracker.update({"left": (LEFT, lanes.SEED_TOP, 1)})
        self.assertNotIn("left", tracker.models)
        tracker.update({"left": (LEFT, lanes.SEED_TOP, 1)})
        self.assertEqual(tracker.status["left"], "detected")

    def test_missing_lanes_fade_identically_on_both_sides(self):
        tracker = lanes.LaneTracker()
        confirm(tracker, left=LEFT, right=RIGHT)
        for missed in range(1, lanes.MAX_MISSING_FRAMES + 1):
            tracker.update({})
            self.assertEqual(tracker.status["left"], "predicted")
            self.assertAlmostEqual(lanes.lane_opacity(tracker, "left"), lanes.lane_opacity(tracker, "right"))
            self.assertEqual(tracker.missing["left"], missed)
        self.assertLess(lanes.lane_opacity(tracker, "left"), 0.5)

    def test_predicted_lane_follows_its_visible_partner_with_a_cap(self):
        tracker = lanes.LaneTracker()
        confirm(tracker, left=LEFT, right=RIGHT)
        start = np.polyval(tracker.models["left"], lanes.DRAW_BOTTOM)
        drifted = RIGHT.copy()
        for step in range(1, 4):
            drifted = drifted + np.array([0.0, 0.0, 0.01])
            tracker.update({"right": (drifted, lanes.SEED_TOP, 1)})
        self.assertEqual(tracker.status["left"], "predicted")
        moved = np.polyval(tracker.models["left"], lanes.DRAW_BOTTOM) - start
        self.assertGreater(moved, 0.015)
        self.assertLess(moved, 3 * lanes.MAX_SHIFT_PER_FRAME + 1e-9)

    def test_implausible_lane_pair_drops_the_weaker_lane(self):
        tracker = lanes.LaneTracker()
        too_close = RIGHT.copy() - np.array([0.0, 0.0, 0.2])  # Only ~0.06 wide at the bottom.
        notes = {}
        kept = lanes.validate_lane_pair(
            {"left": (LEFT, 0.5, 0.9), "right": (too_close, 0.5, 0.6)}, tracker, notes)
        self.assertEqual(list(kept), ["left"])
        self.assertIn("right", notes)

        kept = lanes.validate_lane_pair({"left": (LEFT, 0.5, 0.9), "right": (RIGHT, 0.5, 0.6)}, tracker, {})
        self.assertEqual(set(kept), {"left", "right"})

    def test_held_lane_is_dropped_when_pair_stops_making_sense(self):
        tracker = lanes.LaneTracker()
        confirm(tracker, left=LEFT, right=RIGHT)
        tracker.update({"right": (RIGHT, lanes.SEED_TOP, 1)})
        tracker.models["left"] = RIGHT + np.array([0.0, 0.0, -0.02])  # Nearly on top of the right lane.
        tracker._enforce_pair()
        self.assertNotIn("left", tracker.models)
        self.assertIn("right", tracker.models)

    def test_partner_guide_has_lane_width_at_bottom_and_zero_at_horizon(self):
        guide = lanes.partner_guide(RIGHT, 0.26, "left")
        self.assertAlmostEqual(np.polyval(RIGHT, lanes.DRAW_BOTTOM) - np.polyval(guide, lanes.DRAW_BOTTOM), 0.26)
        self.assertAlmostEqual(np.polyval(RIGHT, lanes.HORIZON_Y), np.polyval(guide, lanes.HORIZON_Y))
        guide = lanes.partner_guide(LEFT, 0.26, "right")
        self.assertAlmostEqual(np.polyval(guide, lanes.DRAW_BOTTOM) - np.polyval(LEFT, lanes.DRAW_BOTTOM), 0.26)

    def test_dash_refresh_nudges_tracked_lane_but_rejects_inconsistent_patches(self):
        y = np.array([0.62, 0.66])
        points = [(value, np.polyval(LEFT, value) + 0.01, 60) for value in y]
        model, top, confidence = lanes.refresh_from_dashes(points, LEFT, 0.55)
        self.assertAlmostEqual(np.polyval(model, 0.7) - np.polyval(LEFT, 0.7), 0.01)
        self.assertEqual(top, 0.55)
        self.assertEqual(confidence, lanes.DASH_REFRESH_CONFIDENCE)

        scattered = [(0.62, np.polyval(LEFT, 0.62) + 0.03, 60), (0.66, np.polyval(LEFT, 0.66) - 0.01, 60)]
        self.assertIsNone(lanes.refresh_from_dashes(scattered, LEFT, 0.55))
        self.assertIsNone(lanes.refresh_from_dashes(points[:1], LEFT, 0.55))

    def test_arch_that_reverses_direction_is_not_a_lane(self):
        mask = np.zeros((720, 1280), dtype=np.uint8)
        y = np.linspace(0.54, 0.70, 7)
        arch = [(value, 0.40 - 2 * (value - 0.62) ** 2, 60) for value in y]
        self.assertIsNone(lanes.fit_lane_curve(mask, "left", LEFT, arch))

    def test_evidence_must_be_continuous_and_near_the_car(self):
        mask = np.zeros((720, 1280), dtype=np.uint8)
        far_only = [(value, np.polyval(LEFT, value), 60) for value in (0.52, 0.55, 0.58)]
        self.assertIsNone(lanes.fit_lane_curve(mask, "left", LEFT, far_only))

        # A gap larger than MAX_POINT_GAP cuts off everything beyond it.
        gappy = [(value, np.polyval(LEFT, value), 60) for value in (0.70, 0.67, 0.64, 0.61, 0.58, 0.43)]
        model, top, _ = lanes.fit_lane_curve(mask, "left", LEFT, gappy)
        self.assertAlmostEqual(top, 0.58)  # The stray far patch past the gap is ignored.

        too_short_after_cut = [(value, np.polyval(LEFT, value), 60) for value in (0.70, 0.67, 0.64, 0.45)]
        self.assertIsNone(lanes.fit_lane_curve(mask, "left", LEFT, too_short_after_cut))

    def test_wrong_orientation_is_rejected(self):
        mask = np.zeros((720, 1280), dtype=np.uint8)
        y = np.linspace(0.54, 0.70, 6)
        right_leaning_on_left_side = [(v, 0.36 + 0.9 * (v - 0.72), 60) for v in y]  # Leans the wrong way.
        self.assertIsNone(lanes.fit_lane_curve(mask, "left", LEFT, right_leaning_on_left_side))

    def test_paint_mask_finds_dim_white_and_pale_yellow_but_not_wide_concrete(self):
        frame = np.full((540, 960, 3), 80, dtype=np.uint8)  # Asphalt.
        cv2.line(frame, (330, 405), (420, 285), (140, 140, 140), 6)  # Dim white stripe.
        cv2.line(frame, (500, 405), (560, 285), (45, 125, 165), 6)  # Yellow stripe.
        cv2.rectangle(frame, (640, 300), (860, 400), (150, 150, 150), -1)  # Wide concrete apron.
        mask = lanes.lane_color_mask(frame)
        self.assertGreater(np.count_nonzero(mask[285:405, 330:425]), 100)
        self.assertGreater(np.count_nonzero(mask[285:405, 495:565]), 100)
        self.assertEqual(np.count_nonzero(mask[310:390, 660:840]), 0)

    def test_noise_does_not_create_lanes(self):
        rng = np.random.default_rng(1)
        tracker = lanes.LaneTracker()
        for _ in range(10):
            lanes.process_frame(rng.integers(0, 256, (540, 960, 3), dtype=np.uint8), tracker)
        self.assertEqual(tracker.models, {})

    def test_both_lanes_share_one_visual_style(self):
        tracker = lanes.LaneTracker()
        confirm(tracker, left=LEFT, right=RIGHT)
        frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
        drawn = lanes.draw_lanes(frame, tracker)
        colors = set()
        for side in lanes.SIDES:
            points = lanes.lane_points(tracker, side, 1280, 720).reshape(-1, 2)
            x, y = points[len(points) // 2]
            colors.add(tuple(drawn[y, x]))
        self.assertEqual(colors, {lanes.LANE_COLOR})

    def test_debug_view_is_separate_from_clean_output(self):
        tracker = lanes.LaneTracker()
        frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
        clean = lanes.process_frame(frame, tracker)
        np.testing.assert_array_equal(clean, frame)  # Nothing detected, nothing drawn.
        debug = lanes.draw_debug(frame, tracker)
        self.assertEqual(debug.shape, frame.shape)
        self.assertGreater(np.count_nonzero(debug != frame), 0)  # ROI outline and status text.
        np.testing.assert_array_equal(clean, frame)


if __name__ == "__main__":
    unittest.main()

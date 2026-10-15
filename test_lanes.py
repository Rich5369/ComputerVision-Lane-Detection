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
        capture.read.side_effect = [(True, frame)] * 100

        with (patch.object(sys, "argv", ["lanes.py"]),
              patch.object(lanes.cv2, "VideoCapture", return_value=capture),
              patch.object(lanes.cv2, "namedWindow"),
              patch.object(lanes.cv2, "resizeWindow"),
              patch.object(lanes.cv2, "imshow") as show,
              patch.object(lanes.cv2, "waitKey", return_value=ord("q")),
              patch.object(lanes.cv2, "destroyAllWindows") as close,
              patch.object(lanes, "process_frame", return_value=frame)):
            lanes.main()

        self.assertEqual(show.call_count, 1)  # Shown once, then Q ends playback.
        self.assertLess(capture.read.call_count, 20)  # The reader thread only reads a few frames ahead, then stops.
        capture.release.assert_called_once()
        close.assert_called_once()

    def test_debug_flag_opens_a_second_window_only_when_requested(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for argv, windows in ((["lanes.py"], 1), (["lanes.py", "--debug"], 2)):
            capture = Mock()
            capture.isOpened.return_value = True
            capture.get.return_value = 30
            capture.read.side_effect = [(True, frame)] * 100
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

    def test_paint_segments_keep_dash_gaps_and_merge_solid_chips(self):
        height, width = 540, 960
        model = np.array([0.0, 0.0, 0.5])  # A vertical lane at x = 0.5 keeps the test simple.
        x = width // 2

        dashed = np.zeros((height, width), dtype=np.uint8)
        for start in (275, 315, 355):  # 20 px dashes separated by 20 px of bare asphalt.
            dashed[start:start + 20, x - 2:x + 3] = 255
        runs = lanes.paint_segments(dashed, model, 0.5)
        self.assertEqual(len(runs), 3)
        for (y0, y1), start in zip(runs, (275, 315, 355)):
            self.assertAlmostEqual(y0 * height, start, delta=4)
            self.assertAlmostEqual(y1 * height, start + 20, delta=4)

        solid = np.zeros((height, width), dtype=np.uint8)
        solid[275:385, x - 2:x + 3] = 255
        solid[320:323, :] = 0  # A 3 px chip in the paint is not a dash gap.
        self.assertEqual(len(lanes.paint_segments(solid, model, 0.5)), 1)

        self.assertEqual(lanes.paint_segments(np.zeros((height, width), dtype=np.uint8), model, 0.5), [])

    def test_blob_filter_removes_squat_and_horizontal_shapes_but_keeps_stripes(self):
        paint = np.zeros((540, 960), dtype=np.uint8)
        cv2.line(paint, (400, 500), (440, 400), 255, 5)  # Steep stripe.
        cv2.rectangle(paint, (600, 400), (640, 420), 255, -1)  # Squat blob (taillight/plate).
        cv2.rectangle(paint, (100, 400), (300, 408), 255, -1)  # Horizontal bar (stop line/bumper).
        kept = lanes.remove_non_stripe_blobs(paint)
        self.assertGreater(np.count_nonzero(kept[390:510, 395:445]), 100)
        self.assertEqual(np.count_nonzero(kept[395:425, 595:645]), 0)
        self.assertEqual(np.count_nonzero(kept[395:410, 95:305]), 0)

    def test_dash_candidate_needs_two_aligned_patches_near_the_car(self):
        seed = LEFT
        good = [(0.68, np.polyval(seed, 0.68), 50), (0.65, np.polyval(seed, 0.65), 50), (0.62, np.polyval(seed, 0.62), 50)]
        model, top, confidence = lanes.fit_dash_candidate(good, "left", seed)
        self.assertAlmostEqual(top, 0.62)
        self.assertLess(confidence, 1.0)
        self.assertIsNone(lanes.fit_dash_candidate(good[:1], "left", seed))  # One patch is not a dash.
        self.assertIsNone(lanes.fit_dash_candidate(good, "right", seed))  # Wrong side of the car.
        far = [(0.56, 0.5, 50), (0.54, 0.5, 50)]
        self.assertIsNone(lanes.fit_dash_candidate(far, "left", seed))  # Nowhere near the car.

    def test_drawn_lane_follows_measured_paint_segments(self):
        tracker = lanes.LaneTracker()
        confirm(tracker, left=LEFT)
        frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
        tracker.segments = {"left": [(0.60, 0.64), (0.69, 0.72)]}
        drawn = lanes.draw_lanes(frame, tracker)
        gap_x = int(np.polyval(LEFT, 0.665) * 1280)
        self.assertTrue(np.array_equal(drawn[int(0.665 * 720), gap_x], frame[0, 0]))  # The gap stays bare.
        tracker.segments = {"left": []}  # Nothing visible this frame: draw nothing.
        np.testing.assert_array_equal(lanes.draw_lanes(frame, tracker), frame)

    def _draw_dashes(self, spans, dashed=False):
        tracker = lanes.LaneTracker()
        confirm(tracker, left=LEFT)
        frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
        tracker.segments = {"left": spans}
        tracker.ends = {"left": [tuple(np.polyval(LEFT, span)) for span in spans]}
        tracker.dashed["left"] = dashed
        return frame, lanes.draw_lanes(frame, tracker)

    def _is_drawn(self, frame, drawn, y):
        x = int(np.polyval(LEFT, y) * 1280)
        return not np.array_equal(drawn[int(y * 720), x], frame[0, 0])

    def test_gap_between_dashes_is_closed_as_firmly_as_the_paint(self):
        frame, drawn = self._draw_dashes([(0.60, 0.64), (0.69, 0.72)])
        self.assertTrue(self._is_drawn(frame, drawn, 0.665))  # Inside the gap.
        gap = np.abs(drawn[int(0.665 * 720), int(np.polyval(LEFT, 0.665) * 1280)].astype(int) - 90).sum()
        paint = np.abs(drawn[int(0.70 * 720), int(np.polyval(LEFT, 0.70) * 1280)].astype(int) - 90).sum()
        self.assertEqual(gap, paint)  # No fainter than the dash next to it.

    def test_wide_gap_between_dashes_is_closed_too(self):
        frame, drawn = self._draw_dashes([(0.52, 0.55), (0.69, 0.72)])  # 0.14 of the frame apart.
        for y in (0.58, 0.62, 0.66):
            self.assertTrue(self._is_drawn(frame, drawn, y), y)

    def test_dashed_lane_is_continued_past_its_outermost_dashes_but_a_solid_one_is_not(self):
        spans = [(0.60, 0.63), (0.66, 0.68)]
        frame, dashed = self._draw_dashes(spans, dashed=True)
        self.assertTrue(self._is_drawn(frame, dashed, 0.71))  # Below the nearest dash, down toward the car.
        frame, solid = self._draw_dashes(spans, dashed=False)
        self.assertFalse(self._is_drawn(frame, solid, 0.71))

    def test_overlay_is_not_drawn_on_a_car_ahead(self):
        tracker = lanes.LaneTracker()
        confirm(tracker, left=LEFT)
        frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
        x = np.polyval(LEFT, 0.62)
        tracker.vehicles = [(x - 0.05, 0.55, x + 0.05, 0.66)]
        drawn = lanes.draw_lanes(frame, tracker)
        row = int(0.62 * 720)
        self.assertTrue(np.array_equal(drawn[row, int(x * 1280)], frame[0, 0]))  # Under the car: bare.
        self.assertFalse(np.array_equal(drawn[int(0.70 * 720), int(np.polyval(LEFT, 0.70) * 1280)], frame[0, 0]))

    def test_vehicles_are_found_from_taillight_pairs_not_from_signals_above_the_horizon(self):
        frame = np.full((540, 960, 3), 90, dtype=np.uint8)
        cv2.rectangle(frame, (430, 300), (442, 310), (30, 30, 230), -1)  # Left taillight (BGR red).
        cv2.rectangle(frame, (500, 300), (512, 310), (30, 30, 230), -1)  # Right taillight.
        cv2.rectangle(frame, (700, 100), (712, 110), (30, 30, 230), -1)  # Red traffic signal in the sky.
        boxes = lanes.find_vehicles(frame)
        self.assertEqual(len(boxes), 1)
        x0, y0, x1, y1 = boxes[0]
        self.assertLess(x0 * 960, 430)
        self.assertGreater(x1 * 960, 512)
        self.assertLess(y0 * 540, 300)
        self.assertGreater(y1 * 540, 310)

    def test_paint_over_grass_is_flagged_but_paint_over_asphalt_is_not(self):
        height, width = 540, 960
        model = np.array([0.0, 0.0, 0.5])
        runs = [(0.55, 0.70)]
        asphalt = np.full((height, width, 3), 80, dtype=np.uint8)
        grass = asphalt.copy()
        grass[:] = (40, 120, 50)  # BGR green.
        self.assertFalse(lanes.on_vegetation(asphalt, model, runs))
        self.assertTrue(lanes.on_vegetation(grass, model, runs))

    def test_double_line_keeps_the_stripe_nearest_the_car(self):
        columns = np.array([100, 101, 102, 110, 111, 112])  # Two stripes, 8 px apart.
        left_keep = lanes.car_side_stripe(columns, "left", 2.0)
        right_keep = lanes.car_side_stripe(columns, "right", 2.0)
        np.testing.assert_array_equal(columns[left_keep], [110, 111, 112])  # Left boundary: the right-hand stripe.
        np.testing.assert_array_equal(columns[right_keep], [100, 101, 102])  # Right boundary: the left-hand stripe.
        single = np.array([100, 101, 102, 103])
        self.assertTrue(lanes.car_side_stripe(single, "left", 2.0).all())  # One stripe is untouched.

    def test_lane_over_a_curb_edge_is_dropped_but_real_paint_is_kept(self):
        height, width = 540, 960
        x = int(np.polyval(LEFT, 0.65) * width)
        road = np.full((height, width), 70, dtype=np.uint8)
        paint = road.copy()
        for row in range(int(0.5 * height), int(0.72 * height)):
            cx = int(np.polyval(LEFT, row / height) * width)
            paint[row, cx - 2:cx + 3] = 170  # Bright stripe on dark asphalt.
        curb = np.full((height, width), 150, dtype=np.uint8)  # Concrete on both sides: no contrast.

        def run(image):
            tracker = lanes.LaneTracker()
            confirm(tracker, left=LEFT)
            tracker.segments = {"left": [(0.55, 0.70)]}
            tracker.ends = {"left": []}
            for _ in range(40):
                lanes.verify_paint(tracker, image)
            return tracker

        self.assertIn("left", run(paint).models)
        self.assertNotIn("left", run(curb).models)

    def test_lane_is_hidden_when_paint_runs_out_and_returns_with_it(self):
        height, width = 540, 960
        tracker = lanes.LaneTracker()
        confirm(tracker, left=LEFT)
        road = np.full((height, width), 70, dtype=np.uint8)
        paint = road.copy()
        for row in range(int(0.5 * height), int(0.72 * height)):
            cx = int(np.polyval(LEFT, row / height) * width)
            paint[row, cx - 2:cx + 3] = 170
        faint = road.copy()
        for row in range(int(0.5 * height), int(0.72 * height)):
            cx = int(np.polyval(LEFT, row / height) * width)
            faint[row, cx - 2:cx + 3] = 90  # 20 levels brighter than the road: weaker than paint, stronger than nothing.

        def step(image):
            tracker.segments = {"left": [(0.55, 0.70)]}
            tracker.ends = {"left": []}
            lanes.verify_paint(tracker, image)

        for _ in range(30):
            step(paint)
        self.assertFalse(tracker.hidden.get("left"))
        for _ in range(lanes.HIDE_AFTER_FRAMES + 40):  # The running average first has to sink from the paint level.
            step(faint)
        self.assertTrue(tracker.hidden.get("left"))
        self.assertEqual(tracker.segments["left"], [])  # Kept, but nothing is drawn.
        self.assertIn("left", tracker.models)
        for _ in range(40):
            step(paint)
        self.assertFalse(tracker.hidden.get("left"))  # Paint is back, so the lane is drawn again.

    def test_paint_crossing_the_lane_is_not_drawn_as_part_of_it(self):
        height, width = 540, 960
        mask = np.zeros((height, width), dtype=np.uint8)
        y0, y1 = 0.60, 0.66
        for row in range(int(y0 * height), int(y1 * height)):
            x = int(0.2 * width + (row - y0 * height) * 7)  # A flat bar sweeping across the lane.
            mask[row, x:x + 5] = 255
        self.assertIsNone(lanes.segment_ends(mask, LEFT, (y0, y1), band=0.4))

    def test_paint_positions_that_pull_in_opposite_directions_are_not_trusted(self):
        runs = [(0.52, 0.56), (0.57, 0.72)]
        on_curve = [tuple(np.polyval(LEFT, run)) for run in runs]
        lagging = [tuple(x + 0.02 for x in pair) for pair in on_curve]  # Same direction everywhere: a lane lagging in a turn.
        mixed = [tuple(x + 0.01 for x in on_curve[0]), tuple(x - 0.01 for x in on_curve[1])]
        self.assertTrue(lanes.consistent_ends(LEFT, runs, lagging))
        self.assertFalse(lanes.consistent_ends(LEFT, runs, mixed))

    def test_lone_boundary_cannot_jump_sideways_without_a_partner(self):
        tracker = lanes.LaneTracker()
        confirm(tracker, right=RIGHT)
        before = tracker.models["right"].copy()
        curb = RIGHT + np.array([0.0, 0.0, 0.09])  # A neighbouring edge, persistently present.
        for _ in range(lanes.REACQUIRE_FRAMES + 2):
            tracker.update({"right": (curb, lanes.SEED_TOP, 1)})
        np.testing.assert_array_equal(tracker.models["right"], before)

    def test_lost_lane_is_not_replaced_far_away_for_a_while(self):
        tracker = lanes.LaneTracker()
        confirm(tracker, right=RIGHT)
        for _ in range(lanes.MAX_MISSING_FRAMES + 1):
            tracker.update({})
        self.assertNotIn("right", tracker.models)
        curb = RIGHT + np.array([0.0, 0.0, 0.12])
        for _ in range(lanes.CONFIRM_FRAMES + 2):
            tracker.update({"right": (curb, lanes.SEED_TOP, 1)})
        self.assertNotIn("right", tracker.models)  # Too far from where the lane was lost.
        for _ in range(lanes.CONFIRM_FRAMES):
            tracker.update({"right": (RIGHT, lanes.SEED_TOP, 1)})
        self.assertIn("right", tracker.models)  # Back where it was: accepted.

    def test_dashed_lanes_are_held_longer_than_solid_ones(self):
        tracker = lanes.LaneTracker()
        confirm(tracker, left=LEFT)
        self.assertEqual(tracker.limit("left"), lanes.MAX_MISSING_FRAMES)
        tracker.dashed["left"] = True
        self.assertEqual(tracker.limit("left"), lanes.MAX_MISSING_FRAMES_DASHED)

    def test_paint_contrast_separates_paint_on_asphalt_from_a_curb_edge(self):
        height, width = 540, 960
        runs = [(0.56, 0.70)]
        model = np.array([0.0, 0.0, 0.5])

        paint = np.full((height, width, 3), 80, dtype=np.uint8)  # Asphalt both sides.
        paint[:, 478:483] = 170
        paint_contrast = lanes.paint_contrast(lanes.smoothed_value(paint), model, runs)

        curb = np.full((height, width, 3), 80, dtype=np.uint8)  # Asphalt on one side, bright concrete on the other.
        curb[:, 478:] = 160
        curb_contrast = lanes.paint_contrast(lanes.smoothed_value(curb), model, runs)

        self.assertGreater(paint_contrast, lanes.MIN_PAINT_CONTRAST)
        self.assertLess(curb_contrast, lanes.MIN_PAINT_CONTRAST_TRACKED)
        self.assertIsNone(lanes.paint_contrast(lanes.smoothed_value(paint), model, []))  # Nothing to measure.

    def test_curb_check_does_not_reject_a_real_painted_line(self):
        frame = np.full((540, 960, 3), 80, dtype=np.uint8)
        cv2.line(frame, (640, 389), (545, 270), (170, 170, 170), 5)  # Right boundary toward the horizon.
        found = lanes.detect_lanes(frame, lanes.LaneTracker())
        self.assertIn("right", found)

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

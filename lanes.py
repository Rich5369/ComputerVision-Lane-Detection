import argparse
import time
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np

# --- Region of interest (fractions of frame width/height) -------------------
ROI_TOP = 0.50  # The horizon sits at 0.47-0.50 in the sample footage; keep skyline out.
ROI_BOTTOM = 0.76
ROI_TOP_LEFT = 0.30
ROI_TOP_RIGHT = 0.70
ROI_BOTTOM_LEFT = 0.08
ROI_BOTTOM_RIGHT = 0.92
DRAW_BOTTOM = 0.72  # Lowest point lanes are drawn/fitted (just above the hood).
SEED_TOP = 0.55
DETECTION_WIDTH = 960

# --- Paint detection ---------------------------------------------------------
TOPHAT_KERNEL = 25  # px at DETECTION_WIDTH; bright shapes narrower than this count as paint.
WHITE_CONTRAST = 22  # How much brighter than the nearby road white paint must be (0-255).
WHITE_MIN_VALUE = 110  # HSV brightness; dim dusk paint is around 130-170.
WHITE_MAX_SATURATION = 60  # HSV saturation; grass and tinted surfaces are higher.
WHITE_MAX_SATURATION_LOOSE = 80  # Fallback-pass limit: dim dusk dashes in a turn reach ~65-75.
YELLOW_HUE = (8, 30)  # OpenCV hue units (0-179).
YELLOW_MIN_SATURATION = 80
YELLOW_MIN_VALUE = 110
YELLOW_CONTRAST = 15
NEIGHBOR_DISTANCE = 9  # px at DETECTION_WIDTH: how far left/right to compare against.
ISOLATION_CONTRAST = 14  # White paint must beat the road on both sides by this much.
MAX_STRIPE_WIDTH = 15 # px at DETECTION_WIDTH; thicker paint-colored shapes are not stripes.
MAX_MASK_DENSITY = 0.12  # If this much of the ROI looks like paint, it is clutter, not a lane.
MIN_BLOB_AREA = 12  # px at DETECTION_WIDTH; smaller specks are noise.
MIN_BLOB_ELONGATION = 2.2  # Long side / short side; paint stripes are long and thin, car parts are squat.
MIN_BLOB_ANGLE = 18  # Degrees from horizontal; flatter blobs are stop bars, bumpers or crosswalk bars.
MAX_FLAT_DASH_LENGTH = 38  # px at DETECTION_WIDTH; short flat slivers are dashes seen at a shallow angle (turns), not bars.
MIN_TURN_BLOB_ANGLE = 8  # Fallback pass: a line sweeping across the view while turning is flatter than usual.
MIN_FLAT_DASH_ANGLE = 3  # Degrees from horizontal; anything flatter is a stop bar or bumper edge.

# --- Curb rejection ----------------------------------------------------------
MIN_PAINT_CONTRAST = 24  # V levels (0-255): a new lane's paint must beat the road on both sides by this much.
MIN_PAINT_CONTRAST_TRACKED = 16  # Lower for a lane already tracked, so it does not flap near the limit.
CONTRAST_NEAR = 9  # px at DETECTION_WIDTH: road is sampled from this far from the line...
CONTRAST_FAR = 16  # ...out to this far, on each side.

# --- Twin stripes (double yellow) ------------------------------------------------
TWIN_GAP = 0.004  # Gap (fraction of frame width) between pixel columns that splits paint into two stripes.

MAX_END_DISAGREEMENT = 0.004  # Corrections of opposite sign larger than this (fraction of width) mean the pieces were measured inconsistently.

# --- Checking that a drawn lane is still on paint ----------------------------
CONTRAST_SMOOTHING = 0.15  # Weight of the newest frame in a lane's running paint contrast.
MIN_TRACKED_CONTRAST = 12  # A lane whose running contrast is below this is a curb, grass or concrete edge: drop it.
HIDE_CONTRAST = 22  # A lane that once showed real paint is kept but not drawn while its contrast is below this...
SHOW_CONTRAST = 28  # ...until it climbs back above this (a gap in the edge line at a driveway, then paint again).
TRUSTED_CONTRAST = 30  # Only a lane that once showed this much contrast may be hidden rather than judged new.
HIDE_AFTER_FRAMES = 15  # ...and only after it has stayed that low this many frames in a row; thin distant dashes dip briefly.
MAX_SLOPE_DEVIATION = 1.0  # Pixel slope (dx/dy) a drawn piece may differ from its lane; crossing paint (stop bars) is far off.

# --- Dash / solid segmentation (what the overlay actually draws) ------------
SEGMENT_BAND = 0.012  # Half-width (fraction of frame width) around the lane curve searched for paint.
SEGMENT_MIN_LENGTH = 0.010  # Shortest painted run drawn (fraction of frame height).
SEGMENT_MERGE_GAP = 0.012  # Gaps up to this (fraction of height) inside paint are treated as chips, not dash gaps.
SEGMENT_BAND_DASHED = 0.024  # Wider search for dashed lanes: while turning, dashes sweep sideways faster than the model follows.
GAP_OPACITY = 0.55  # Fill between dashes is drawn fainter than the paint itself.
MAX_GAP_FILL = 0.12  # Longest gap (fraction of frame height) bridged between two dashes.
SEGMENT_HOLD_FRAMES = 6  # A tracked lane keeps its last drawn dashes this long when none are seen (mask flicker, passing car).
SEGMENT_PAD = 0.003  # Extend each run slightly so overlay ends sit on the paint edges.

# --- Vehicles ahead (taillight clusters) -------------------------------------
VEHICLE_TOP = 0.48  # Lights above this are traffic signals and signs, not cars.
VEHICLE_BOTTOM = 0.72  # Lights below this are reflections on the hood.
TAILLIGHT_SATURATION = 120
TAILLIGHT_VALUE = 140
TAILLIGHT_MIN_PIXELS = 4
MAX_TAILLIGHT_SIZE = 0.05  # Longest side of one light (fraction of frame width).
MAX_CAR_WIDTH = 0.3  # Widest car box estimated from lights (fraction of frame width).
VEHICLE_MARGIN = 0.15  # Extra box size around a red body, as a fraction of its width.
MAX_VEGETATION = 8  # Green dominance (G minus the larger of R, B) at the line above which it sits on grass.

# --- Searching along a lane --------------------------------------------------
WINDOW_COUNT = 10
WINDOW_MARGIN = 0.055  # Fraction of the detection frame width.
MIN_WINDOW_PIXELS = 12
MAX_WINDOW_PIXELS = 500  # Per 1280px width; dense blobs are not narrow stripes.
MAX_SEED_SPAN = 0.18  # Reject lines that lean across too much of the road.
HOUGH_VOTES = 18  # Paint pixels a straight seed needs (px at DETECTION_WIDTH).
HOUGH_MIN_LENGTH = 18
HOUGH_MAX_GAP = 60  # Wide enough to bridge the gap between dashes.
HORIZON_Y = 0.49  # Where lanes converge; used to scale lane width with distance.

# --- Geometric validation ----------------------------------------------------
MAX_POINT_GAP = 0.14  # Largest vertical gap (fraction of height) the evidence may skip.
NEAREST_EVIDENCE = 0.58  # Evidence must reach at least this far down the image.
MIN_LANE_SLOPE = 0.2  # |dx/dy| in pixels: a lane may not be closer to vertical than this...
MAX_LANE_SLOPE = 2.8  # ...or closer to horizontal than this.
MAX_FIT_RMS = 0.012  # Typical distance of marking patches from the curve (fraction of width).
OUTLIER_DISTANCE = 0.02
MIN_FIT_CONFIDENCE = 0.45
MAX_BEND = 0.08  # Quadratic bend limit; larger means an unrealistic curve.
LANE_WIDTH_MIN = 0.14  # Left-to-right distance at DRAW_BOTTOM (fraction of frame width).
LANE_WIDTH_MAX = 0.40
LANE_WIDTH_TOLERANCE = 0.35  # Allowed change from the recently tracked lane width.

# --- Tracking ----------------------------------------------------------------
MAX_MISSING_FRAMES = 8  # How long a lane may be predicted without new evidence.
DASHED_PAINTED_FRACTION = 0.7  # A lane with less paint than this along its length counts as dashed.
MAX_MISSING_FRAMES_DASHED = 36  # Dashed lanes are legitimately unseen between dashes (~1 s); only visible paint is drawn.
MIN_DRAW_CONFIDENCE = 0.1
SMOOTHING = 0.35
CURVE_SMOOTHING = 0.5  # Curvature is smoothed this much more slowly than position.
TOP_SMOOTHING = 0.2  # Slow smoothing keeps the far endpoint from jittering.
MAX_FRAME_JUMP = 0.06  # Largest one-frame lane shift accepted (fraction of width).
RELOCK_FRAMES = 90  # After losing a lane, how long a far-away replacement is distrusted (~3 s).
RELOCK_DISTANCE = 0.05  # How far (fraction of width) a lone replacement may sit from the lost lane.
REACQUIRE_FRAMES = 5  # Frames a "jumped" lane must stay put before we believe it.
CONFIRM_FRAMES = 3  # Consecutive detections before a brand-new lane is shown.
ACQUIRE_CONFIDENCE = 0.6  # A new lane must score this well; an already-tracked one only needs MIN_FIT_CONFIDENCE.
DASH_REFRESH_CONFIDENCE = 0.5  # Confidence when a tracked lane is re-anchored on a dash.
DASH_REFRESH_MIN_SPAN = 0.03  # Vertical extent (fraction of height) of a usable dash.
DASH_CANDIDATE_CONFIDENCE = 0.62  # Slightly above ACQUIRE_CONFIDENCE, but only used when both sides agree.
MAX_SHIFT_PER_FRAME = 0.02  # Cap when a predicted lane follows its partner sideways.

_SAMPLE_Y = np.array([ROI_TOP, (ROI_TOP + DRAW_BOTTOM) / 2, DRAW_BOTTOM])  # Where two lane models are compared.

LANE_COLOR = (255, 160, 0)
LANE_OUTLINE = (20, 20, 20)
LINE_WIDTH = 4  # Overlay thickness in px at 1280 wide (scaled with the frame).
OUTLINE_OPACITY = 0.35  # How strongly the dark edge shows (0 = none).
SIDES = ("left", "right")


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0] * DRAW_BOTTOM)
    y2 = int(image.shape[0] * SEED_TOP)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def find_lane_seeds(image, lines, loose=False):
    """Pick one straight Hough segment cluster per side as a starting guide."""
    height, width = image.shape[:2]
    fits = {"left": [], "right": []}
    if lines is None:
        return {}
    # While turning, the line ahead sweeps across the view: allow shallower, wider seeds.
    min_slope = 0.18 if loose else 0.35
    span_limit = 0.3 if loose else MAX_SEED_SPAN
    left_top = (0.3, 0.7) if loose else (0.3, 0.55)
    right_top = (0.3, 0.7) if loose else (0.45, 0.7)

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)
        if not min_slope < abs(slope) < 5:
            continue
        intercept = y1 - slope * x1
        bottom_x = (height * DRAW_BOTTOM - intercept) / slope
        top_x = (height * SEED_TOP - intercept) / slope
        length = np.hypot(x2 - x1, y2 - y1)

        if (slope < 0 and 0.25 * width < bottom_x < 0.5 * width
                and left_top[0] * width < top_x < left_top[1] * width
                and top_x - bottom_x < span_limit * width):
            fits["left"].append((bottom_x, slope, intercept, length))
        elif (slope > 0 and 0.5 * width < bottom_x < 0.84 * width
              and right_top[0] * width < top_x < right_top[1] * width
              and bottom_x - top_x < span_limit * width):
            fits["right"].append((bottom_x, slope, intercept, length))

    selected_lines = {}
    radius = width * 0.035
    for side in SIDES:
        candidates = fits[side]
        if not candidates:
            continue

        supports = [
            sum(other[3] for other in candidates if abs(other[0] - fit[0]) < radius)
            for fit in candidates
        ]
        strongest = max(supports)
        viable = [fit for fit, support in zip(candidates, supports) if support >= 0.6 * strongest]
        best = min(
            viable,
            key=lambda fit: abs(fit[0] - width / 2),
        )
        cluster = [fit for fit in candidates if abs(fit[0] - best[0]) < radius]
        parameters = np.average(
            [(fit[1], fit[2]) for fit in cluster],
            axis=0,
            weights=[fit[3] for fit in cluster],
        )
        selected_lines[side] = make_coordinates(image, parameters)

    if "left" in selected_lines and "right" in selected_lines:
        left, right = selected_lines["left"], selected_lines["right"]
        if left[2] >= right[2]:
            wrong_side = max(
                SIDES,
                key=lambda candidate_side: abs(selected_lines[candidate_side][2] - width / 2),
            )
            del selected_lines[wrong_side]

    return selected_lines


def roi_polygon(width, height):
    return np.array([
        [
            (int(width * ROI_BOTTOM_LEFT), int(height * ROI_BOTTOM)),
            (int(width * ROI_BOTTOM_RIGHT), int(height * ROI_BOTTOM)),
            (int(width * ROI_TOP_RIGHT), int(height * ROI_TOP)),
            (int(width * ROI_TOP_LEFT), int(height * ROI_TOP)),
        ]
    ], dtype=np.int32)


def region_of_interest(image):
    height, width = image.shape[:2]
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, roi_polygon(width, height), 255)
    return cv2.bitwise_and(image, mask)


def line_model(line, width, height):
    """Convert a Hough seed to x/width = a*y/height + b."""
    x_bottom, y_bottom, x_top, y_top = line
    slope = ((x_bottom - x_top) / width) / ((y_bottom - y_top) / height)
    return np.array([0.0, slope, x_bottom / width - slope * y_bottom / height])


def lane_color_mask(frame, loose=False):
    """Mark pixels that look like narrow white or yellow road paint.

    White is judged against the road around it (a morphological top-hat), not
    against a fixed brightness, so dim dusk paint still counts while large
    bright things such as cars, sky and concrete do not.
    """
    height, width = frame.shape[:2]
    top, bottom = int(height * ROI_TOP), int(height * ROI_BOTTOM) + 1
    hsv = cv2.cvtColor(np.ascontiguousarray(frame[top:bottom]), cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    scale = width / DETECTION_WIDTH
    size = max(9, round(TOPHAT_KERNEL * scale)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    contrast = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)

    # Real paint has darker road on BOTH sides. A curb lip or gutter edge is
    # bright against asphalt on one side but concrete (equally bright) on the other.
    smooth = cv2.GaussianBlur(value, (3, 3), 0).astype(np.int16)
    reach = max(3, round(NEIGHBOR_DISTANCE * scale))
    brighter_than_both = smooth - np.maximum(np.roll(smooth, reach, axis=1), np.roll(smooth, -reach, axis=1))

    white = ((contrast > WHITE_CONTRAST) & (value > WHITE_MIN_VALUE) & (saturation < (WHITE_MAX_SATURATION_LOOSE if loose else WHITE_MAX_SATURATION))
             & (brighter_than_both > ISOLATION_CONTRAST))
    yellow = (
        (hue >= YELLOW_HUE[0]) & (hue <= YELLOW_HUE[1])
        & (saturation >= YELLOW_MIN_SATURATION) & (value >= YELLOW_MIN_VALUE)
        & (contrast > YELLOW_CONTRAST)
    )
    paint = np.zeros((height, width), dtype=np.uint8)
    paint[top:bottom] = (white | yellow).astype(np.uint8) * 255
    paint = region_of_interest(paint)

    # Cars, crosswalk bars and pale concrete are thick; stripes are thin. Erase
    # anything thicker than a stripe so only line-like paint remains.
    thick_size = max(5, round(MAX_STRIPE_WIDTH * scale)) | 1
    thick_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thick_size, thick_size))
    core = cv2.erode(paint, thick_kernel)
    if core.any():
        grown = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thick_size + 8, thick_size + 8)))
        paint[grown > 0] = 0
    return remove_non_stripe_blobs(paint, allow_flat_dashes=loose)


def remove_non_stripe_blobs(paint, allow_flat_dashes=False):
    """Keep only connected paint blobs shaped like a lane stripe.

    The color/contrast tests only ask "is this pixel bright?". Taillights, plates,
    bumpers and stop bars pass that test, so we also look at each blob's SHAPE:
    a lane stripe is long and thin and runs away from the camera (steep in the
    image), while a car part is squat or horizontal. The shape comes from the
    blob's second moments (a covariance ellipse), which is cheap and needs no
    contour tracing.
    """
    scale = paint.shape[1] / DETECTION_WIDTH
    count, labels, stats, _ = cv2.connectedComponentsWithStats(paint, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < MIN_BLOB_AREA * scale ** 2:
            continue
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == label)
        if xs.size < 3:
            continue
        covariance = np.cov(np.vstack((xs, ys)))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        long_axis = eigenvectors[:, 1]
        elongation = np.sqrt(max(eigenvalues[1], 1e-6) / max(eigenvalues[0], 1e-6))
        angle_from_horizontal = np.degrees(np.arctan2(abs(long_axis[1]), abs(long_axis[0])))
        # While turning, far dashes sweep sideways and look flat but stay short; bars are long.
        short_flat_dash = (allow_flat_dashes and max(w, h) <= MAX_FLAT_DASH_LENGTH * scale
                           and angle_from_horizontal >= MIN_FLAT_DASH_ANGLE)
        min_angle = MIN_TURN_BLOB_ANGLE if allow_flat_dashes else MIN_BLOB_ANGLE
        keep[label] = elongation >= MIN_BLOB_ELONGATION and (
            angle_from_horizontal >= min_angle or short_flat_dash)
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def paint_segments(mask, model, top, bottom=None, band=None):
    """Find the painted runs along a lane curve: one run for solid paint, several for dashes.

    Idea: walk down the lane curve row by row and ask "is there paint right on
    the curve in this row?". That gives a 1-D on/off signal. Runs of "on" are
    paint; runs of "off" are bare asphalt. Tiny "off" holes (chips in a solid
    line) are filled; real dash gaps are longer and stay gaps.

    Returns a list of (y_start, y_end) in fractions of frame height, top first.
    """
    bottom = DRAW_BOTTOM if bottom is None else bottom
    height, width = mask.shape[:2]
    rows = np.arange(int(top * height), int(bottom * height) + 1)
    if rows.size == 0:
        return []
    curve_x = np.polyval(model, rows / height) * width
    band = (SEGMENT_BAND if band is None else band) * width
    on_curve = np.zeros(rows.size, dtype=bool)
    for index, (row, x) in enumerate(zip(rows, curve_x)):
        left, right = max(0, int(x - band)), min(width, int(x + band) + 1)
        on_curve[index] = left < right and mask[min(row, height - 1), left:right].any()

    # Run-length encode the on/off signal.
    padded = np.concatenate(([False], on_curve, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    runs = [[rows[start], rows[end - 1]] for start, end in zip(changes[::2], changes[1::2])]

    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= SEGMENT_MERGE_GAP * height:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    pad = SEGMENT_PAD * height
    return [
        (max(top, (start - pad) / height), min(bottom, (end + pad) / height))
        for start, end in merged if (end - start) >= SEGMENT_MIN_LENGTH * height
    ]


def segment_ends(mask, model, run, band=SEGMENT_BAND_DASHED, side=None):
    """Where one painted run really sits: its x (fraction of width) at the top and bottom of the run.

    The tracked curve smooths over several frames, so in a turn it lags the paint.
    Drawing on the paint itself keeps each dash exactly on its stripe. Falls back
    to the curve when too little paint is found or the paint is far from it, and
    returns None for a run that does not point along the lane at all.
    """
    height, width = mask.shape[:2]
    rows = np.arange(int(run[0] * height), int(run[1] * height) + 1)
    curve_x = np.polyval(model, rows / height) * width
    reach = band * width
    ys, xs = [], []
    for row, x in zip(rows, curve_x):
        left, right = max(0, int(x - reach)), min(width, int(x + reach) + 1)
        found = np.flatnonzero(mask[min(row, height - 1), left:right]) + left
        found = found[car_side_stripe(found, side, max(2.0, TWIN_GAP * width))]
        if found.size:
            ys.append(row)
            xs.append(float(np.median(found)))
    fallback = tuple(float(np.polyval(model, y)) for y in run)
    if len(ys) < 3:
        return fallback
    slope, intercept = np.polyfit(ys, xs, 1)
    if len(ys) >= 6:
        # Paint that crosses the lane (a stop bar, a crosswalk edge) runs in another direction.
        lane_slope = float(np.polyval(np.polyder(model), (run[0] + run[1]) / 2)) * width / height
        if abs(slope - lane_slope) > MAX_SLOPE_DEVIATION:
            return None
    ends = tuple(float((slope * y * height + intercept) / width) for y in run)
    if max(abs(end - curve) for end, curve in zip(ends, fallback)) > band:
        return fallback
    return ends


def car_side_stripe(xs, side, gap):
    """Boolean mask over `xs` keeping, of two stripes side by side (a double line), the one nearest the car.

    A left boundary is the right-hand stripe of a pair and a right boundary the
    left-hand one. `xs` are pixel columns of paint; the pair is told apart by a gap
    between columns wider than a stripe is thick. A single stripe keeps everything.
    """
    keep = np.ones(xs.shape, dtype=bool)
    if side is None or xs.size < 2:
        return keep
    columns = np.unique(xs)
    jumps = np.flatnonzero(np.diff(columns) > gap)
    if jumps.size == 0:
        return keep
    if side == "left":
        return xs > columns[jumps[-1]]
    return xs <= columns[jumps[0]]


def collect_lane_points(mask, guide, loose=False, side=None):
    """Follow a guide upward and keep the center of each narrow marking patch."""
    height, width = mask.shape[:2]
    pixel_y, pixel_x = np.nonzero(mask)
    window_height = (DRAW_BOTTOM - ROI_TOP) / WINDOW_COUNT
    margin = (0.085 if loose else WINDOW_MARGIN) * width
    offset = 0.0
    points = []

    for index in range(WINDOW_COUNT):
        top = DRAW_BOTTOM - (index + 1) * window_height
        bottom = top + window_height
        middle = (top + bottom) / 2
        predicted_x = np.polyval(guide, middle) * width + offset
        inside = (
            (pixel_y >= top * height) & (pixel_y < bottom * height)
            & (np.abs(pixel_x - predicted_x) < margin)
        )
        nearby_x = pixel_x[inside]
        inside_rows = pixel_y[inside]
        keep = car_side_stripe(nearby_x, side, max(2.0, TWIN_GAP * width))
        if keep.sum() >= MIN_WINDOW_PIXELS:
            nearby_x, inside_rows = nearby_x[keep], inside_rows[keep]
        if nearby_x.size < MIN_WINDOW_PIXELS:
            continue
        # Broad blobs are usually cars or sidewalks, not a narrow lane stripe.
        if nearby_x.size > MAX_WINDOW_PIXELS * (width / 1280) ** 2:
            continue
        if np.percentile(nearby_x, 75) - np.percentile(nearby_x, 25) > margin * 0.9:
            continue
        observed_x = float(np.median(nearby_x))
        observed_y = float(np.median(inside_rows))
        points.append((observed_y / height, observed_x / width, min(nearby_x.size, 100)))
        offset = 0.65 * (observed_x - np.polyval(guide, middle) * width)
    return points


def contiguous_from_bottom(points):
    """Keep only evidence that is continuous upward from the nearest patch.

    A lane may skip a dash, but a long empty stretch means whatever lies beyond
    it is unrelated (or an intersection), so it must not stretch the lane.
    """
    ordered = sorted(points, key=lambda point: -point[0])
    run = []
    for point in ordered:
        if run and run[-1][0] - point[0] > MAX_POINT_GAP:
            break
        run.append(point)
    return run


def fit_lane_curve(mask, side, guide, points=None, notes=None, loose=False):
    """Fit a straight or quadratic boundary through observed marking patches.

    Returns (model, top, confidence) or None. When `notes` is a dict, the reason
    for a rejection is stored in notes[side] for the debug view.
    """
    def reject(reason):
        if notes is not None:
            notes[side] = reason
        return None

    height, width = mask.shape[:2]
    aspect = width / height
    if points is None:
        points = collect_lane_points(mask, guide, loose, side)
    points = contiguous_from_bottom(points)

    if len(points) < 3:
        return reject("too few marking patches")
    if max(point[0] for point in points) - min(point[0] for point in points) < 0.09:
        return reject("evidence too short")
    if points[0][0] < NEAREST_EVIDENCE:
        return reject("no evidence near the car")

    y_values = np.array([point[0] for point in points])
    x_values = np.array([point[1] for point in points])
    weights = np.sqrt([point[2] for point in points])
    straight = np.polyfit(y_values, x_values, 1, w=weights)

    # Drop patches far from the first straight estimate (a stray blob), refit once.
    residual = np.abs(x_values - np.polyval(straight, y_values))
    inliers = residual <= OUTLIER_DISTANCE
    if 3 <= inliers.sum() < len(points):
        y_values, x_values, weights = y_values[inliers], x_values[inliers], weights[inliers]
        straight = np.polyfit(y_values, x_values, 1, w=weights)
    if len(y_values) < 3 or np.ptp(y_values) < 0.09:
        return reject("outliers removed too much evidence")
    straight_model = np.array([0.0, *straight])

    # A curve needs several separated observations and must explain the pixels
    # substantially better than a straight line, or straight roads would wobble.
    model = straight_model
    if len(y_values) >= 5 and np.ptp(y_values) >= 0.16:
        curved = np.polyfit(y_values, x_values, 2, w=weights)
        straight_error = np.average((x_values - np.polyval(straight_model, y_values)) ** 2, weights=weights)
        curved_error = np.average((x_values - np.polyval(curved, y_values)) ** 2, weights=weights)
        bend = abs(curved[0]) * (DRAW_BOTTOM - ROI_TOP) ** 2 / 4
        slopes = np.polyval(np.polyder(curved), [ROI_TOP, DRAW_BOTTOM])
        bends_one_way = slopes[0] * slopes[1] > 0  # No hooks or S-shapes.
        bottom_shift = abs(np.polyval(curved, DRAW_BOTTOM) - np.polyval(straight_model, DRAW_BOTTOM))
        if (curved_error < straight_error * 0.75 and 0.004 < bend < MAX_BEND
                and bends_one_way and bottom_shift < 0.03):
            model = curved

    rms = float(np.sqrt(np.average((x_values - np.polyval(model, y_values)) ** 2, weights=weights)))
    if rms > MAX_FIT_RMS:
        return reject(f"fit too loose ({rms:.3f})")

    top_x, bottom_x = np.polyval(model, [ROI_TOP, DRAW_BOTTOM])
    reach = 0.22 if loose else 0.0
    if side == "left":
        valid = 0.2 < top_x < 0.64 + reach and 0.12 < bottom_x < 0.52 and bottom_x < top_x - 0.015
    else:
        valid = 0.36 - reach < top_x < 0.8 and 0.48 < bottom_x < 0.9 and bottom_x > top_x + 0.015
    if not valid or abs(top_x - bottom_x) > 0.38 + reach:
        return reject("outside the road area")
    if abs(bottom_x - np.polyval(guide, DRAW_BOTTOM)) > 0.04:
        return reject("drifted from guide")

    # Orientation in pixels at the middle of the evidence: left lanes lean
    # right going up, right lanes lean left, and neither may be near-flat/vertical.
    middle_y = float(np.mean(y_values))
    pixel_slope = np.polyval(np.polyder(model), middle_y) * aspect
    max_slope = 5.5 if loose else MAX_LANE_SLOPE
    if side == "left":
        oriented = -max_slope < pixel_slope < -MIN_LANE_SLOPE
    else:
        oriented = MIN_LANE_SLOPE < pixel_slope < max_slope
    if not oriented:
        return reject(f"bad orientation ({pixel_slope:.2f})")

    count_score = min(1.0, len(y_values) / 6)
    span_score = min(1.0, float(np.ptp(y_values)) / 0.2)
    fit_score = max(0.0, 1 - rms / MAX_FIT_RMS)
    reach_score = float(np.clip(1 - (DRAW_BOTTOM - np.max(y_values)) / MAX_POINT_GAP, 0, 1))
    confidence = 0.3 * count_score + 0.25 * span_score + 0.25 * fit_score + 0.2 * reach_score
    if confidence < MIN_FIT_CONFIDENCE:
        return reject(f"low confidence ({confidence:.2f})")
    return model, float(np.min(y_values)), float(confidence)


def smoothed_value(frame):
    """Brightness (HSV 'V') of the frame, lightly blurred so single noisy pixels don't count."""
    blue, green, red = cv2.split(frame)
    return cv2.GaussianBlur(cv2.max(cv2.max(blue, green), red), (5, 5), 0)


def paint_contrast(value, model, runs):
    """How much brighter the painted runs are than the road on their LEFT and RIGHT, or None if unmeasurable.

    Real paint is a bright stripe on darker asphalt, so it beats the pixels on
    both sides. A curb edge or sidewalk joint passes the paint-pixel tests but
    is about as bright as the concrete next to it, so one side (or both) has
    almost no contrast. The weaker side decides, so a line must stand out
    everywhere. Distances are measured in pixels at DETECTION_WIDTH.
    """
    height, width = value.shape[:2]
    scale = width / DETECTION_WIDTH
    rows = np.concatenate([np.arange(int(start * height), int(end * height)) for start, end in runs]) if runs else []
    if len(rows) == 0:
        return None
    rows = np.asarray(rows)
    x = (np.polyval(model, rows / height) * width).astype(int)
    peak_window = np.ones((1, max(3, round(13 * scale)) | 1), dtype=np.uint8)
    peak = cv2.dilate(value, peak_window)  # Brightest pixel within a few px: tolerates small model error.
    inside = (x > 0) & (x < width - 1)
    rows, x = rows[inside], x[inside]
    if len(rows) < 8:
        return None
    line = peak[rows, x].mean()
    offsets = np.arange(round(CONTRAST_NEAR * scale), round(CONTRAST_FAR * scale), 2)
    sides = []
    for direction in (-1, 1):
        columns = np.clip(x[:, None] + direction * offsets[None, :], 0, width - 1)
        # Darkest ring, not the average: the twin of a double yellow line is bright
        # but the asphalt between and beyond the pair is not.
        sides.append(value[rows[:, None], columns].mean(axis=0).min())
    return float(line - max(sides))


def on_vegetation(frame, model, runs):
    """True when the painted runs sit on grass: green clearly beats red and blue right at the line."""
    height, width = frame.shape[:2]
    if not runs:
        return False
    rows = np.concatenate([np.arange(int(start * height), int(end * height)) for start, end in runs])
    x = (np.polyval(model, rows / height) * width).astype(int)
    inside = (x > 4) & (x < width - 5)
    rows, x = rows[inside], x[inside]
    if len(rows) < 8:
        return False
    reach = max(2, round(4 * width / DETECTION_WIDTH))
    columns = x[:, None] + np.arange(-reach, reach + 1)[None, :]
    pixels = frame[rows[:, None], columns].astype(np.int16)
    greenness = pixels[..., 1] - np.maximum(pixels[..., 0], pixels[..., 2])
    return float(greenness.mean()) > MAX_VEGETATION


def find_vehicles(frame):
    """Boxes (x0, y0, x1, y1 as fractions) for cars ahead, found from their red taillights.

    Each bright saturated red blob is a light. Two lights at about the same height
    and a car-width apart make one car; a lone light gets a box sized from the
    light itself. A big solid red patch is a red car and uses its own bounds. Only
    the road band is searched, so traffic signals and signs above the horizon and
    hood reflections below are ignored.
    """
    height, width = frame.shape[:2]
    top, bottom = int(height * VEHICLE_TOP), int(height * VEHICLE_BOTTOM)
    hue, saturation, value = cv2.split(cv2.cvtColor(np.ascontiguousarray(frame[top:bottom]), cv2.COLOR_BGR2HSV))
    red = (((hue <= 9) | (hue >= 168)) & (saturation >= TAILLIGHT_SATURATION)
           & (value >= TAILLIGHT_VALUE)).astype(np.uint8) * 255
    if not red.any():
        return []
    scale = width / DETECTION_WIDTH
    count, _, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
    boxes, lights = [], []
    for x, y, w, h, area in stats[1:]:
        if area < TAILLIGHT_MIN_PIXELS * scale ** 2:
            continue
        y += top
        if area > 0.0015 * width * height and area > 0.35 * w * h:  # A red body.
            pad = VEHICLE_MARGIN * w / 2
            boxes.append((x - pad, y - pad, x + w + pad, y + h + pad))
        elif max(w, h) <= MAX_TAILLIGHT_SIZE * width:  # Larger non-solid red is paint, a sign or part of a body.
            lights.append((x, y, w, h))

    def car_box(x0, y0, x1, y1):
        car_width = min(max((x1 - x0) * 1.2, 10 * scale), MAX_CAR_WIDTH * width)
        centre = (x0 + x1) / 2
        return centre - 0.5 * car_width, y0 - 0.25 * car_width, centre + 0.5 * car_width, y1 + 0.5 * car_width

    lights.sort()
    used = set()
    for i, (xa, ya, wa, ha) in enumerate(lights):
        if i in used:
            continue
        for j in range(i + 1, len(lights)):
            xb, yb, wb, hb = lights[j]
            spacing = xb - (xa + wa)
            if j in used or spacing > 0.1 * width:
                continue
            similar = max(wa * ha, wb * hb) <= 3 * min(wa * ha, wb * hb)  # Both lights of one car look alike.
            if similar and spacing >= max(wa, wb) and abs((ya + ha / 2) - (yb + hb / 2)) <= 0.3 * (spacing + wa + wb):
                boxes.append(car_box(xa, min(ya, yb), xb + wb, max(ya + ha, yb + hb)))
                used.update((i, j))
                break
    for i, (x, y, w, h) in enumerate(lights):
        if i not in used and not any(x >= bx0 and x + w <= bx1 and y >= by0 and y + h <= by1
                                     for bx0, by0, bx1, by1 in boxes):
            centre = x + w / 2
            half = max(2.5 * w, 6 * scale)
            boxes.append(car_box(centre - half, y, centre + half, y + h))
    return [(max(0.0, x0 / width), max(0.0, y0 / height), min(1.0, x1 / width), min(1.0, y1 / height))
            for x0, y0, x1, y1 in boxes]


def side_geometry_ok(side, model):
    """Is this curve where a lane boundary on `side` of the car can plausibly be, and oriented like one?"""
    top_x, bottom_x = np.polyval(model, [ROI_TOP, DRAW_BOTTOM])
    if side == "left":
        located = 0.2 < top_x < 0.64 and 0.12 < bottom_x < 0.52 and bottom_x < top_x - 0.015
    else:
        located = 0.36 < top_x < 0.8 and 0.48 < bottom_x < 0.9 and bottom_x > top_x + 0.015
    return located and abs(top_x - bottom_x) <= 0.38


def fit_dash_candidate(points, side, seed_model):
    """Weak boundary evidence from one or two dashes, for lanes marked only by dashes.

    A single dash is too short to fit a trustworthy curve, so this uses the
    straight Hough seed and only checks that the dash's patches sit on it. The
    result is flagged weaker than a normal fit, and detect_lanes only keeps it
    when the opposite boundary is also found and the two form a believable lane.
    """
    points = contiguous_from_bottom(points)
    if len(points) < 2:
        return None
    y_values = np.array([point[0] for point in points])
    if np.ptp(y_values) < DASH_REFRESH_MIN_SPAN or y_values.max() < NEAREST_EVIDENCE:
        return None
    offsets = np.array([point[1] for point in points]) - np.polyval(seed_model, y_values)
    if np.max(np.abs(offsets)) > 0.012 or not side_geometry_ok(side, seed_model):
        return None
    return seed_model, float(y_values.min()), DASH_CANDIDATE_CONFIDENCE


def refresh_from_dashes(points, model, top):
    """Re-anchor a tracked lane on a dash or two without changing its shape.

    One dash is too short to fit a line from scratch, but it can confirm that an
    already-tracked lane is still where we think it is and nudge it sideways.
    """
    if len(points) < 2:
        return None
    y_values = np.array([point[0] for point in points])
    offsets = np.array([point[1] for point in points]) - np.polyval(model, y_values)
    if (y_values.max() < NEAREST_EVIDENCE - 0.02 or np.ptp(y_values) < DASH_REFRESH_MIN_SPAN
            or np.ptp(offsets) > 0.012 or abs(np.median(offsets)) > 0.03):
        return None
    return model + np.array([0.0, 0.0, float(np.median(offsets))]), top, DASH_REFRESH_CONFIDENCE


def validate_lane_pair(candidates, tracker, notes):
    """Drop the weaker lane when the left/right pair is not a believable lane."""
    if "left" not in candidates or "right" not in candidates:
        return candidates

    left, right = candidates["left"], candidates["right"]
    width_bottom = np.polyval(right[0], DRAW_BOTTOM) - np.polyval(left[0], DRAW_BOTTOM)
    shared_top = max(left[1], right[1])
    sample_y = np.linspace(shared_top, DRAW_BOTTOM, 5)
    gaps = np.polyval(right[0], sample_y) - np.polyval(left[0], sample_y)

    problem = None
    if not LANE_WIDTH_MIN <= width_bottom <= LANE_WIDTH_MAX:
        problem = f"lane width {width_bottom:.2f}"
    elif np.any(gaps <= 0.01):
        problem = "lanes cross"
    elif gaps[0] > gaps[-1] * 1.05:
        problem = "lanes diverge into the distance"
    elif (tracker.lane_width is not None
          and abs(width_bottom - tracker.lane_width) > LANE_WIDTH_TOLERANCE * tracker.lane_width):
        problem = f"lane width {width_bottom:.2f} vs tracked {tracker.lane_width:.2f}"
    if problem is None:
        return candidates

    # Keep the lane with more support; prefer the one already being tracked.
    def support(side):
        tracked = side in tracker.models
        return candidates[side][2] + (0.25 if tracked else 0.0)

    weaker = min(SIDES, key=support)
    notes[weaker] = f"pair rejected: {problem}"
    return {side: value for side, value in candidates.items() if side != weaker}


class LaneTracker:
    """Track each lane independently: detected, predicted (held), or gone."""

    def __init__(self):
        self.models: dict[str, np.ndarray] = {}
        self.tops: dict[str, float] = {}
        self.missing: dict[str, int] = {}
        self.confidence: dict[str, float] = {}
        self.status: dict[str, str] = {}
        self.lane_width = None
        self.debug: dict = {}
        # Painted runs per side measured on the current frame: [(y_start, y_end), ...].
        # A side with no entry was not measured, so drawing falls back to the whole curve.
        self.segments: dict[str, list[tuple[float, float]]] = {}
        self.contrast: dict[str, float] = {}  # Running paint contrast of each tracked lane.
        self.contrast_peak: dict[str, float] = {}  # What that contrast was while the lane was most trusted.
        self.low_frames: dict[str, int] = {}  # Consecutive checks a lane's contrast has stayed below HIDE_CONTRAST.
        self.hidden: dict[str, bool] = {}  # Lane kept but not drawn while its paint is far below its trusted level.
        self.unseen: dict[str, int] = {}  # Consecutive frames a lane showed no paint but kept its last dashes.
        self.ends: dict[str, list[tuple[float, float]]] = {}  # x at each run's (top, bottom), parallel to segments.
        self.frame = 0
        self.vehicles: list[tuple[float, float, float, float]] = []  # Boxes of cars ahead, as fractions.
        self.last_seen: dict[str, tuple[float, int]] = {}  # side -> (bottom x, frame); survives _forget.
        self.dashed: dict[str, bool] = {}  # Sticky: set once a lane is seen as separate dashes.
        self._peak: dict[str, float] = {}
        self._pending: dict[str, tuple[np.ndarray, int]] = {}
        self._candidate: dict[str, tuple[np.ndarray, int]] = {}

    def limit(self, side):
        """Frames a lane may go without evidence before it is dropped."""
        return MAX_MISSING_FRAMES_DASHED if self.dashed.get(side) else MAX_MISSING_FRAMES

    def fade(self, side):
        """1.0 while a lane is detected; falls toward 0 as a held lane runs out of time."""
        if self.status.get(side) == "detected":
            return 1.0
        return self.confidence[side] / max(self._peak[side], 1e-6)

    def _forget(self, side):
        for store in (self.models, self.tops, self.missing, self.confidence,
                      self.status, self.dashed, self._peak, self._pending, self._candidate,
                      self.contrast, self.contrast_peak, self.hidden, self.low_frames):
            store.pop(side, None)

    def _confirm(self, side, model, confidence):
        """A brand-new lane is shown only after consecutive, consistent, strong detections."""
        if confidence < ACQUIRE_CONFIDENCE:
            self._candidate.pop(side, None)
            return False
        pending = self._candidate.get(side)
        if pending is not None and np.max(np.abs(np.polyval(pending[0], _SAMPLE_Y) - np.polyval(model, _SAMPLE_Y))) <= MAX_FRAME_JUMP:
            count = pending[1] + 1
        else:
            count = 1
        self._candidate[side] = (model, count)
        return count >= CONFIRM_FRAMES

    def _enforce_pair(self):
        """If the held/detected pair stops looking like one lane, drop the weaker side."""
        if not all(side in self.models for side in SIDES):
            return
        sample_y = np.linspace(max(self.tops.values()), DRAW_BOTTOM, 5)
        gaps = np.polyval(self.models["right"], sample_y) - np.polyval(self.models["left"], sample_y)
        if LANE_WIDTH_MIN <= gaps[-1] <= LANE_WIDTH_MAX and np.all(gaps > 0.01):
            return
        predicted = [side for side in SIDES if self.status[side] == "predicted"]
        weaker = predicted[0] if len(predicted) == 1 else min(SIDES, key=lambda side: self.confidence[side])
        self._forget(weaker)

    def _width_plausible(self, side, model):
        """Would `model` keep the lane about as wide as it has been, given the other boundary?"""
        partner = self.models.get("right" if side == "left" else "left")
        if partner is None or self.lane_width is None:
            return True
        gap = np.polyval(model, DRAW_BOTTOM) - np.polyval(partner, DRAW_BOTTOM)
        width = gap if side == "right" else -gap
        return abs(width - self.lane_width) <= LANE_WIDTH_TOLERANCE * self.lane_width

    def _has_partner(self, side):
        return ("right" if side == "left" else "left") in self.models

    def _near_last_position(self, side, model):
        """A lane lost moments ago may not reappear far away unless the other boundary is present.

        Without this, a lane that fades at an intersection is replaced by whatever
        straight edge (curb, sidewalk) shows up next. Real paint comes back where it was.
        """
        last = self.last_seen.get(side)
        if last is None or self.frame - last[1] > RELOCK_FRAMES or self._has_partner(side):
            return True
        return abs(np.polyval(model, DRAW_BOTTOM) - last[0]) <= RELOCK_DISTANCE

    def _accept_jump(self, side, model):
        """A big shift is believed only after it repeats consistently."""
        pending = self._pending.get(side)
        if pending is not None and np.max(np.abs(np.polyval(pending[0], _SAMPLE_Y) - np.polyval(model, _SAMPLE_Y))) < 0.04:
            count = pending[1] + 1
        else:
            count = 1
        self._pending[side] = (np.asarray(model), count)
        return count >= REACQUIRE_FRAMES

    def update(self, detections):
        self.frame += 1
        bottoms_before = {side: np.polyval(model, DRAW_BOTTOM) for side, model in self.models.items()}
        detected = set()

        for side in SIDES:
            result = detections.get(side)
            model, top, confidence = result if result is not None else (None, None, 0.0)
            previous = self.models.get(side)
            reacquired = False

            if model is not None and previous is not None:
                shift = np.max(np.abs(np.polyval(model, _SAMPLE_Y) - np.polyval(previous, _SAMPLE_Y)))
                if shift > MAX_FRAME_JUMP:
                    # Discard one-frame jumps from passing cars or road edges,
                    # unless the new position persists (a real lane change).
                    if not self._has_partner(side) or not self._width_plausible(side, model):
                        # A lone boundary that leaps sideways is more likely a curb or a
                        # neighbouring lane than a lane change, and with no partner there is
                        # nothing to check it against. A partner makes the lane suddenly
                        # wider/narrower when it is the wrong boundary.
                        self._pending.pop(side, None)
                        model = None
                    elif self._accept_jump(side, model):
                        self._forget(side)
                        previous = None
                        reacquired = True
                    else:
                        model = None
                else:
                    self._pending.pop(side, None)

            if model is not None and previous is None and not reacquired:
                if not self._near_last_position(side, model) or not self._confirm(side, np.asarray(model, dtype=float), confidence):
                    continue
            elif model is None and previous is None:
                self._candidate.pop(side, None)

            if model is None:
                if previous is not None:
                    self.missing[side] += 1
                    if self.missing[side] > self.limit(side):
                        self._forget(side)
                    else:
                        fade = 1 - self.missing[side] / (self.limit(side) + 1)
                        self.confidence[side] = self._peak[side] * fade
                        self.status[side] = "predicted"
                continue

            model = np.asarray(model, dtype=float)
            alpha = SMOOTHING * (0.5 + 0.5 * confidence)
            if previous is None:
                self.models[side] = model
                self.tops[side] = top
                self._peak[side] = confidence
            else:
                self.models[side] = blend_models(previous, model, alpha)
                self.tops[side] = (1 - TOP_SMOOTHING) * self.tops[side] + TOP_SMOOTHING * top
                self._peak[side] = 0.5 * self._peak[side] + 0.5 * confidence
            self.missing[side] = 0
            self.confidence[side] = self._peak[side]
            self.status[side] = "detected"
            self.last_seen[side] = (float(np.polyval(self.models[side], DRAW_BOTTOM)), self.frame)
            detected.add(side)

        # A held lane follows its partner when the car drifts sideways.
        for side in SIDES:
            partner = "right" if side == "left" else "left"
            if (self.status.get(side) == "predicted" and partner in detected
                    and partner in bottoms_before):
                moved = np.polyval(self.models[partner], DRAW_BOTTOM) - bottoms_before[partner]
                self.models[side] = self.models[side] + np.array(
                    [0.0, 0.0, float(np.clip(moved, -MAX_SHIFT_PER_FRAME, MAX_SHIFT_PER_FRAME))]
                )

        self._enforce_pair()
        if "left" in detected and "right" in detected and all(side in self.models for side in SIDES):
            width = np.polyval(self.models["right"], DRAW_BOTTOM) - np.polyval(self.models["left"], DRAW_BOTTOM)
            self.lane_width = width if self.lane_width is None else 0.9 * self.lane_width + 0.1 * width


def blend_models(previous, new, alpha):
    """Smooth position at alpha and curvature more slowly, keeping a clean parabola.

    The curve is anchored through the smoothed positions at the middle and
    bottom of the lane; only the bend (coefficient a) gets extra damping.
    """
    middle = (ROI_TOP + DRAW_BOTTOM) / 2
    smoothed = {
        y: (1 - alpha) * np.polyval(previous, y) + alpha * np.polyval(new, y)
        for y in (middle, DRAW_BOTTOM)
    }
    curve_alpha = alpha * CURVE_SMOOTHING
    a = (1 - curve_alpha) * previous[0] + curve_alpha * new[0]
    b = (smoothed[DRAW_BOTTOM] - smoothed[middle] - a * (DRAW_BOTTOM ** 2 - middle ** 2)) / (DRAW_BOTTOM - middle)
    c = smoothed[DRAW_BOTTOM] - a * DRAW_BOTTOM ** 2 - b * DRAW_BOTTOM
    return np.array([a, b, c])


def partner_guide(partner_model, lane_width, side):
    """Expected position of a lane from its neighbour; width shrinks toward the horizon."""
    # x_side(y) = x_partner(y) + direction * k * (y - HORIZON_Y), zero width at the horizon.
    k = lane_width / (DRAW_BOTTOM - HORIZON_Y)
    direction = -1 if side == "left" else 1
    a, b, c = partner_model
    return np.array([a, b + direction * k, c - direction * k * HORIZON_Y])


def detect_lanes(frame, tracker):
    """Find lane candidates, trying progressively more forgiving settings until something fits.

    1. Strict paint mask and normal road geometry.
    2. Looser paint mask (dim dusk dashes in a turn).
    3. Looser mask and wider geometry, for a line sweeping across the view in a bend.
    Each later pass picks up more clutter, so it only runs when the one before found no lane.
    """
    tracker.vehicles = find_vehicles(frame)  # Used to keep the overlay off cars ahead.
    candidates = _detect_with_mask(frame, tracker, np.asarray(lane_color_mask(frame)))
    if not candidates:
        loose_mask = np.asarray(lane_color_mask(frame, loose=True))
        candidates = _detect_with_mask(frame, tracker, loose_mask)
        if not candidates:
            candidates = _detect_with_mask(frame, tracker, loose_mask, loose=True)
    return candidates


def _detect_with_mask(frame, tracker, mask, loose=False):
    height, width = frame.shape[:2]
    roi_area = cv2.contourArea(roi_polygon(width, height))
    density = np.count_nonzero(mask) / roi_area
    notes: dict[str, str] = {}
    debug = {"size": (width, height), "mask": mask, "seeds": {}, "points": {},
             "candidates": {}, "notes": notes, "density": density}
    tracker.debug = debug

    if density > MAX_MASK_DENSITY:
        notes["frame"] = f"marking clutter ({density:.2f})"
        return {}

    scale = width / DETECTION_WIDTH
    segments = cv2.HoughLinesP(
        mask, 2, np.pi / 180, max(10, round(HOUGH_VOTES * scale)), np.array([]),
        minLineLength=max(10, round(HOUGH_MIN_LENGTH * scale)), maxLineGap=max(5, round(HOUGH_MAX_GAP * scale)),
    )
    seeds = find_lane_seeds(frame, segments, loose)
    debug["seeds"] = {side: seed / np.array([width, height, width, height]) for side, seed in seeds.items()}

    candidates = {}
    dash_only = {}
    for side in SIDES:
        seed = seeds.get(side)
        previous = tracker.models.get(side)
        seed_model = line_model(seed, width, height) if seed is not None else None
        guides = []
        if previous is not None:
            guides.append(previous)
        if seed_model is not None:
            if previous is None or abs(np.polyval(seed_model, DRAW_BOTTOM) - np.polyval(previous, DRAW_BOTTOM)) > 0.1:
                guides.insert(0, seed_model)
            else:
                guides.append(seed_model)  # Second chance if the tracked guide finds nothing.
        partner = "right" if side == "left" else "left"
        if not guides and partner in tracker.models and tracker.lane_width is not None:
            # No seed here, but the other lane and the usual lane width say where to look.
            # This only steers the search; a line still needs real paint evidence.
            guides.append(partner_guide(tracker.models[partner], tracker.lane_width, side))
        if not guides:
            notes[side] = "no guide"
            continue

        for guide in guides:
            points = collect_lane_points(mask, guide, loose, side)
            debug["points"][side] = points
            curve = fit_lane_curve(mask, side, guide, points, notes, loose)
            if curve is not None:
                notes.pop(side, None)
                candidates[side] = curve
                break
            if seed_model is not None and side not in dash_only:
                dash = fit_dash_candidate(points, side, seed_model)
                if dash is not None:
                    dash_only[side] = dash

        if side not in candidates and previous is not None:
            points = collect_lane_points(mask, previous, side=side)
            refreshed = refresh_from_dashes(points, previous, tracker.tops[side])
            if refreshed is not None:
                debug["points"][side] = points
                notes.pop(side, None)
                candidates[side] = refreshed

    # Curb rejection: paint must stand out from the road on BOTH sides of it.
    value = smoothed_value(frame)
    for group in (candidates, dash_only):
        for side in list(group):
            model, top, _ = group[side]
            runs = paint_segments(mask, model, top)
            if on_vegetation(frame, model, runs):
                notes[side] = "on grass"
                del group[side]
                continue
            contrast = paint_contrast(value, model, runs)
            needed = MIN_PAINT_CONTRAST_TRACKED if side in tracker.models else MIN_PAINT_CONTRAST
            if contrast is not None and contrast < needed:
                notes[side] = f"not paint-like (contrast {contrast:.0f})"
                del group[side]

    # A dash-only boundary is believed only as one half of a plausible lane pair.
    used_dash = set()
    if len(candidates) + len(dash_only.keys() - candidates.keys()) == 2:
        for side, dash in dash_only.items():
            if side not in candidates:
                candidates[side] = dash
                used_dash.add(side)
    candidates = validate_lane_pair(candidates, tracker, notes)
    if len(candidates) < 2:
        candidates = {side: value for side, value in candidates.items() if side not in used_dash}
    debug["candidates"] = candidates
    return candidates


def lane_opacity(tracker, side):
    """Detected lanes are solid; held lanes fade with their remaining confidence."""
    if tracker.status.get(side) == "detected":
        return 1.0
    return 0.3 + 0.7 * tracker.fade(side)


def lane_spans(tracker, side):
    """The pieces of a lane to draw: painted runs if measured, else one full-length piece."""
    if side in tracker.segments:
        return tracker.segments[side]
    return [None]


def lane_points(tracker, side, width, height, span=None, ends=None):
    """Pixel polyline for a lane, or for one painted run of it when span=(y0, y1) is given.

    `ends` = (x at y0, x at y1) is where the paint really is at the run's ends: the
    curve is nudged sideways to meet them, so a long curved run keeps its bend.
    """
    if span is None:
        top = tracker.tops[side]
        if tracker.status.get(side) != "detected":
            top = DRAW_BOTTOM - (DRAW_BOTTOM - top) * (0.4 + 0.6 * tracker.fade(side))
        span = (top, DRAW_BOTTOM)
    y = np.linspace(span[0], span[1], 40)
    x = np.polyval(tracker.models[side], y)
    if ends is not None:
        correction = np.array(ends) - np.polyval(tracker.models[side], span)
        x = x + np.interp(y, span, correction)
    return np.column_stack((x * width, y * height)).astype(np.int32).reshape(-1, 1, 2)


def outside_vehicles(points, vehicles, width, height):
    """Split a polyline into the pieces that are not inside any car's box."""
    if not vehicles:
        return [points]
    xy = points.reshape(-1, 2)
    inside = np.zeros(len(xy), dtype=bool)
    for x0, y0, x1, y1 in vehicles:
        inside |= ((xy[:, 0] >= x0 * width) & (xy[:, 0] <= x1 * width)
                   & (xy[:, 1] >= y0 * height) & (xy[:, 1] <= y1 * height))
    pieces, start = [], None
    for index, hidden in enumerate(np.append(inside, True)):
        if not hidden and start is None:
            start = index
        elif hidden and start is not None:
            if index - start >= 2:
                pieces.append(points[start:index])
            start = None
    return pieces


def draw_lanes(frame, tracker):
    """Blue paint-following overlay with a faint dark edge so it reads on light and dark road."""
    output = frame.copy()
    height, width = frame.shape[:2]
    scale = width / 1280
    jobs = []  # (points, opacity) for every painted piece to draw
    for side in SIDES:
        if side not in tracker.models or tracker.confidence[side] < MIN_DRAW_CONFIDENCE:
            continue
        opacity = lane_opacity(tracker, side)
        spans = lane_spans(tracker, side)
        ends = tracker.ends.get(side)
        if ends is not None and len(ends) != len(spans):
            ends = None
        for index, span in enumerate(spans):
            span_ends = ends[index] if ends is not None else None
            jobs.append((lane_points(tracker, side, width, height, span, span_ends), opacity))
            # Bridge the gap to the next dash so the lane reads as one continuous line.
            if ends is not None and index + 1 < len(spans) and span is not None:
                gap = (span[1], spans[index + 1][0])
                if 0 < gap[1] - gap[0] <= MAX_GAP_FILL:
                    jobs.append((lane_points(tracker, side, width, height, gap, (ends[index][1], ends[index + 1][0])),
                                 opacity * GAP_OPACITY))
    jobs = [(piece, opacity) for points, opacity in jobs
            for piece in outside_vehicles(points, tracker.vehicles, width, height)]
    if not jobs:
        return output

    # Work only inside the band of rows the overlay touches: blending a whole
    # 4K frame just to soften a few thin lines would dominate the frame time.
    margin = max(2, round(LINE_WIDTH * scale)) + 4
    rows = np.concatenate([points[:, 0, 1] for points, _ in jobs])
    top, bottom = max(0, int(rows.min()) - margin), min(height, int(rows.max()) + margin)
    region = output[top:bottom]
    jobs = [(points - np.array([0, top], dtype=np.int32), opacity) for points, opacity in jobs]

    # Outline first, blended in lightly: a hard black border looks heavy and
    # hides the paint it is meant to highlight.
    shadow = region.copy()
    for points, _ in jobs:
        cv2.polylines(shadow, [points], False, LANE_OUTLINE, max(2, round(LINE_WIDTH * scale)) + 2, cv2.LINE_AA)
    cv2.addWeighted(shadow, OUTLINE_OPACITY, region, 1 - OUTLINE_OPACITY, 0, region)

    for points, opacity in jobs:
        layer = region if opacity >= 0.99 else region.copy()
        cv2.polylines(layer, [points], False, LANE_COLOR, max(2, round(LINE_WIDTH * scale)), cv2.LINE_AA)
        if layer is not region:
            cv2.addWeighted(layer, opacity, region, 1 - opacity, 0, region)
    return output


def draw_debug(frame, tracker):
    """Separate diagnostic view; never mixed into the clean output."""
    output = frame.copy()
    info = tracker.debug
    if not info:
        return output
    height, width = frame.shape[:2]
    scale = width / 1280

    paint = cv2.resize(info["mask"], (width, height), interpolation=cv2.INTER_NEAREST) > 0
    output[paint] = (0.3 * output[paint] + 0.7 * np.array((0, 220, 0))).astype(np.uint8)
    cv2.polylines(output, roi_polygon(width, height), True, (0, 255, 255), max(1, round(2 * scale)))

    for seed in info["seeds"].values():
        x1, y1, x2, y2 = (seed * np.array([width, height, width, height])).astype(int)
        cv2.line(output, (x1, y1), (x2, y2), (255, 255, 255), max(1, round(2 * scale)), cv2.LINE_AA)
    for points in info["points"].values():
        for y, x, _ in points:
            cv2.circle(output, (int(x * width), int(y * height)), max(3, round(6 * scale)), (0, 0, 255), -1)

    for side, (model, top, _) in info["candidates"].items():
        y = np.linspace(top, DRAW_BOTTOM, 30)
        curve = np.column_stack((np.polyval(model, y) * width, y * height)).astype(np.int32)
        cv2.polylines(output, [curve.reshape(-1, 1, 2)], False, (255, 0, 255), max(1, round(3 * scale)), cv2.LINE_AA)

    for x0, y0, x1, y1 in tracker.vehicles:
        cv2.rectangle(output, (int(x0 * width), int(y0 * height)), (int(x1 * width), int(y1 * height)),
                      (0, 0, 255), max(1, round(2 * scale)))

    status_colors = {"detected": (0, 255, 0), "predicted": (0, 165, 255)}
    font_scale = 0.9 * scale
    lines = []
    for side in SIDES:
        if side in tracker.models:
            status = tracker.status[side]
            text = f"{side.upper()}: {status.upper()} conf {tracker.confidence[side]:.2f}"
            if status == "predicted":
                text += f" ({tracker.missing[side]}/{tracker.limit(side)})"
            lines.append((text, status_colors[status]))
            y = np.linspace(tracker.tops[side], DRAW_BOTTOM, 30)
            curve = np.column_stack((np.polyval(tracker.models[side], y) * width, y * height)).astype(np.int32)
            cv2.polylines(output, [curve.reshape(-1, 1, 2)], False, status_colors[status], max(2, round(4 * scale)), cv2.LINE_AA)
        else:
            reason = info["notes"].get(side, "no evidence")
            lines.append((f"{side.upper()}: NONE - {reason}", (0, 0, 255)))
    lines.append((f"paint density {info['density']:.3f}"
                  + (f" - {info['notes']['frame']}" if "frame" in info["notes"] else ""), (255, 255, 255)))
    for index, (text, color) in enumerate(lines):
        position = (int(20 * scale), int((40 + 40 * index) * scale))
        cv2.putText(output, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(3, round(5 * scale)), cv2.LINE_AA)
        cv2.putText(output, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(1, round(2 * scale)), cv2.LINE_AA)
    return output


def consistent_ends(model, runs, ends):
    """Are the per-piece paint positions a believable sideways shift of the lane curve?

    A lane that lags the paint in a turn is off by the same direction all along.
    Pieces that pull in opposite directions were measured on different things
    (the near and far half of a double line, a neighbouring stripe), so none is trusted.
    """
    shifts = [end - float(np.polyval(model, y)) for run, pair in zip(runs, ends) for y, end in zip(run, pair)]
    return not (shifts and max(shifts) > MAX_END_DISAGREEMENT and min(shifts) < -MAX_END_DISAGREEMENT)


def measure_segments(tracker):
    """Record, for each tracked lane, where paint is actually visible in this frame."""
    mask = tracker.debug.get("mask")
    previous_segments, previous_ends = tracker.segments, tracker.ends
    tracker.segments = {}
    tracker.ends = {}
    if mask is None:
        return
    for side, model in tracker.models.items():
        runs = paint_segments(mask, model, ROI_TOP,
                              band=SEGMENT_BAND_DASHED if tracker.dashed.get(side) else None)
        ends = [segment_ends(mask, model, run, side=side) for run in runs]
        runs = [run for run, end in zip(runs, ends) if end is not None]
        ends = [end for end in ends if end is not None]
        if not consistent_ends(model, runs, ends):
            ends = [tuple(float(np.polyval(model, y)) for y in run) for run in runs]
        tracker.segments[side] = runs
        tracker.ends[side] = ends
        if runs:
            tracker.unseen[side] = 0
        elif previous_segments.get(side) and tracker.unseen.get(side, 0) < SEGMENT_HOLD_FRAMES:
            # Nothing visible for a moment: keep the last dashes instead of blinking out.
            tracker.unseen[side] = tracker.unseen.get(side, 0) + 1
            tracker.segments[side] = previous_segments[side]
            tracker.ends[side] = previous_ends.get(side, [])
        # Mostly-unpainted lane length means dashes (or occlusion): hold the lane longer.
        painted = sum(end - start for start, end in runs)
        if len(runs) >= 2 or (runs and painted < DASHED_PAINTED_FRACTION * (DRAW_BOTTOM - tracker.tops[side])):
            tracker.dashed[side] = True


def verify_paint(tracker, value):
    """Drop or hide lanes that are not on real paint.

    Paint is brighter than the road on BOTH sides. A curb face, the edge of a
    concrete apron or grass is not, and measures near zero. A single frame is noisy
    (cars, shadows, thin distant dashes), so each lane keeps a running average. A
    lane near zero is dropped. A lane that was on real paint but stays low for a
    while (the edge line runs out at a driveway) is kept but not drawn until the
    paint returns.
    """
    for side in list(tracker.models):
        model = tracker.models[side]
        contrast = paint_contrast(value, model, tracker.segments.get(side, []))
        if contrast is None:
            continue
        average = tracker.contrast.get(side)
        average = contrast if average is None else (1 - CONTRAST_SMOOTHING) * average + CONTRAST_SMOOTHING * contrast
        tracker.contrast[side] = average
        peak = max(average, tracker.contrast_peak.get(side, average))
        tracker.contrast_peak[side] = peak
        if peak >= TRUSTED_CONTRAST:
            if average >= SHOW_CONTRAST:
                tracker.low_frames[side] = 0
                tracker.hidden[side] = False
            elif average < HIDE_CONTRAST:
                tracker.low_frames[side] = tracker.low_frames.get(side, 0) + 1
                if tracker.low_frames[side] >= HIDE_AFTER_FRAMES:
                    tracker.hidden[side] = True
        if average < MIN_TRACKED_CONTRAST:
            tracker._forget(side)
            tracker.segments.pop(side, None)
            tracker.ends.pop(side, None)
            tracker.unseen.pop(side, None)
        elif tracker.hidden.get(side):
            tracker.segments[side] = []
            tracker.ends[side] = []


def drop_lanes_on_vegetation(tracker, frame):
    """A held or drifting lane can end up over grass; a new candidate is checked for that, so check these too."""
    for side in list(tracker.models):
        if on_vegetation(frame, tracker.models[side], tracker.segments.get(side, [])):
            tracker._forget(side)
            tracker.segments.pop(side, None)
            tracker.ends.pop(side, None)
            tracker.unseen.pop(side, None)


def process_frame(frame, tracker=None):
    if tracker is None:
        tracker = LaneTracker()
    height, width = frame.shape[:2]
    working_frame = (
        np.asarray(cv2.resize(frame, (DETECTION_WIDTH, round(height * DETECTION_WIDTH / width))))
        if width > DETECTION_WIDTH else frame
    )
    tracker.update(detect_lanes(working_frame, tracker))
    measure_segments(tracker)
    verify_paint(tracker, smoothed_value(working_frame))
    drop_lanes_on_vegetation(tracker, working_frame)
    # Models use fractions of width/height, so drawing restores original coordinates.
    return draw_lanes(frame, tracker)


def choose_video():
    root = Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilename(
            title="Select a driving video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Draw detected lane lines on a driving video.")
    parser.add_argument("video", nargs="?", help="Path to a video; opens a file picker if omitted")
    parser.add_argument("--camera", type=int, help="Camera number, usually 0")
    parser.add_argument("--debug", action="store_true",
                        help="Also show a separate window with ROI, paint pixels, fits and confidence")
    args = parser.parse_args()

    if args.camera is not None and args.video is not None:
        parser.error("Provide a video path or --camera, not both")

    default_video = Path(__file__).with_name("test_driving_video.mp4")
    source = (
        args.camera if args.camera is not None
        else args.video or (str(default_video) if default_video.is_file() else choose_video())
    )
    if not source:
        print("No video selected.")
        return

    if isinstance(source, str) and not Path(source).is_file():
        raise FileNotFoundError(f"Could not find video: {source}")

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_interval = 1 / fps if fps > 0 else 1 / 30
    tracker = LaneTracker()
    window_name = "Lane detection - Q or Esc to quit"
    debug_name = "Lane debug"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        if args.debug:
            cv2.namedWindow(debug_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(debug_name, 1280, 720)
        while True:
            frame_start = time.perf_counter()
            success, frame = capture.read()
            if not success:
                break

            frame = np.asarray(frame)
            cv2.imshow(window_name, process_frame(frame, tracker))
            if args.debug:
                cv2.imshow(debug_name, draw_debug(frame, tracker))
            delay = max(1, round((frame_interval - (time.perf_counter() - frame_start)) * 1000))
            if cv2.waitKey(delay) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

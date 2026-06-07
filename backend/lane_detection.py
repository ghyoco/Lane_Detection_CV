"""
Lane detection pipeline — ported from model/lane_detection.ipynb (Cells 3, 4, 6, 8).

Same BEV polynomial-fit approach, day/night configs, sliding-window clustering,
EMA smoothing and drift HUD as the notebook. The only change is that `mode`
("auto" | "day" | "night") is passed explicitly instead of read from an
ipywidgets toggle, so this can run headless inside a web server.
"""

import cv2
import numpy as np

# ── ROI trapezoid — fractions of (width, height), copied from notebook Cell 3 ──
ROI_TOP_LEFT = (0.40, 0.46)
ROI_TOP_RIGHT = (0.62, 0.46)
ROI_BOTTOM_LEFT = (0.12, 0.76)
ROI_BOTTOM_RIGHT = (0.92, 0.76)

# ── Sliding-window & fit constants ──────────────────────────────────────────
N_WINDOWS = 20
MIN_PIX_WIN = 20
MIN_PTS_FIT = 30
DRIFT_THRESHOLD_FRAC = 0.08

# ── Per-condition parameter sets — copied verbatim from notebook Cell 3 ─────
DAY_CFG = {
    'white_l_min': 190,
    'yellow_h': (10, 45),
    'yellow_s_min': 100,
    'canny_sigma': 0.33,
    'window_margin': 80,
    'ema_alpha': 0.30,
    'clahe_clip': 2.0,
    'gamma': 1.0,
    'dilate_iter': 1,
}
NIGHT_CFG = {
    'white_l_min': 110,
    'yellow_h': (8, 50),
    'yellow_s_min': 55,
    'canny_sigma': 0.18,
    'window_margin': 90,
    'ema_alpha': 0.40,
    'clahe_clip': 4.5,
    'gamma': 0.50,
    'dilate_iter': 3,
}


def get_cfg(is_night):
    return NIGHT_CFG if is_night else DAY_CFG


def is_night_frame(gray, mode):
    """`mode` is 'auto' | 'day' | 'night' (case-insensitive)."""
    mode = mode.lower()
    if mode == 'night':
        return True
    if mode == 'day':
        return False
    return float(np.mean(gray)) < 70  # auto


def apply_gamma(gray, gamma):
    if gamma == 1.0:
        return gray
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0
    return lut.astype(np.uint8)[gray]


def adaptive_preprocess(frame, cfg):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_gc = apply_gamma(gray, cfg['gamma'])
    clahe = cv2.createCLAHE(clipLimit=cfg['clahe_clip'], tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_gc)
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    wl, yh, ys = cfg['white_l_min'], cfg['yellow_h'], cfg['yellow_s_min']
    white_mask = cv2.inRange(hls, np.array([0, wl, 0]), np.array([255, 255, 255]))
    yellow_mask = cv2.inRange(hls, np.array([yh[0], 0, ys]), np.array([yh[1], 255, 255]))
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)

    if cfg['dilate_iter'] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        color_mask = cv2.dilate(color_mask, k, iterations=cfg['dilate_iter'])

    return filtered, color_mask


def gaussian_blur(img, kernel=5):
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def adaptive_canny(blurred, sigma):
    v = np.median(blurred)
    lo = int(max(0, (1 - sigma) * v))
    hi = int(min(255, (1 + sigma) * v))
    return cv2.Canny(blurred, lo, hi)


def get_perspective_matrices(shape):
    h, w = shape[:2]
    src = np.float32([
        [w * ROI_BOTTOM_LEFT[0], h * ROI_BOTTOM_LEFT[1]],
        [w * ROI_TOP_LEFT[0], h * ROI_TOP_LEFT[1]],
        [w * ROI_TOP_RIGHT[0], h * ROI_TOP_RIGHT[1]],
        [w * ROI_BOTTOM_RIGHT[0], h * ROI_BOTTOM_RIGHT[1]],
    ])
    dst = np.float32([
        [w * 0.25, h],
        [w * 0.25, 0],
        [w * 0.75, 0],
        [w * 0.75, h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def apply_roi_mask(img, shape):
    h, w = shape[:2]
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    pts = np.array([[
        [int(w * ROI_BOTTOM_LEFT[0]), int(h * ROI_BOTTOM_LEFT[1])],
        [int(w * ROI_TOP_LEFT[0]), int(h * ROI_TOP_LEFT[1])],
        [int(w * ROI_TOP_RIGHT[0]), int(h * ROI_TOP_RIGHT[1])],
        [int(w * ROI_BOTTOM_RIGHT[0]), int(h * ROI_BOTTOM_RIGHT[1])],
    ]], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    return cv2.bitwise_and(img, mask)


def sliding_window_cluster(warped_edges, window_margin):
    h, w = warped_edges.shape
    histogram = np.sum(warped_edges[h // 2:, :], axis=0)
    mid = w // 2
    left_base = int(np.argmax(histogram[:mid]))
    right_base = int(np.argmax(histogram[mid:])) + mid

    # Pinch guard: only reset to fixed defaults when BOTH histogram peaks are
    # genuinely absent (< 15 px). Curves can legitimately bring lanes close
    # together in BEV, so distance alone is not a reliable trigger.
    if (right_base - left_base) < int(w * 0.25):
        if histogram[left_base] < 15 or histogram[right_base] < 15:
            left_base, right_base = int(w * 0.25), int(w * 0.75)

    window_h = h // N_WINDOWS
    nzy, nzx = warped_edges.nonzero()
    nzy, nzx = np.array(nzy), np.array(nzx)

    left_inds, right_inds = [], []
    lx, rx = left_base, right_base

    for win in range(N_WINDOWS):
        y_lo = h - (win + 1) * window_h
        y_hi = h - win * window_h
        xl_lo = max(0, lx - window_margin)
        xl_hi = min(w, lx + window_margin)
        xr_lo = max(0, rx - window_margin)
        xr_hi = min(w, rx + window_margin)

        gl = np.where((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl_lo) & (nzx < xl_hi))[0]
        gr = np.where((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xr_lo) & (nzx < xr_hi))[0]
        left_inds.append(gl)
        right_inds.append(gr)
        if len(gl) >= MIN_PIX_WIN:
            lx = int(np.mean(nzx[gl]))
        if len(gr) >= MIN_PIX_WIN:
            rx = int(np.mean(nzx[gr]))

    li = np.concatenate(left_inds) if left_inds else np.array([], dtype=int)
    ri = np.concatenate(right_inds) if right_inds else np.array([], dtype=int)

    lpts = (nzx[li], nzy[li]) if len(li) else (np.array([]), np.array([]))
    rpts = (nzx[ri], nzy[ri]) if len(ri) else (np.array([]), np.array([]))
    return lpts, rpts


def fit_polynomial(lane_pts):
    xs, ys = lane_pts
    if len(xs) < MIN_PTS_FIT:
        return None
    try:
        coeffs = np.polyfit(ys, xs, 2)
        # Reject only physically impossible curvature (|a| > 0.005 ~ radius < 200px).
        # Do NOT zero small 'a' values — that killed curve detection in the original.
        if abs(coeffs[0]) > 0.005:
            return None
        return coeffs
    except Exception:
        return None


class LaneSmoother:
    """Per-frame exponential moving average over polynomial coefficients.

    Alpha is passed at update-time so day and night can use different speeds
    without needing separate smoother instances.
    """

    def __init__(self):
        self.l = self.r = None

    def update(self, nl, nr, alpha):
        def ema(prev, cur):
            if cur is None:
                return prev
            if prev is None:
                return cur
            return alpha * cur + (1 - alpha) * prev

        self.l = ema(self.l, nl)
        self.r = ema(self.r, nr)
        return self.l, self.r


def detect_drift(bev_shape, lf, rf):
    h, w = bev_shape[:2]
    car_cx = w / 2.0
    y_ev = float(h - 1)
    if lf is None or rf is None:
        return False, 'UNKNOWN', 0.0
    lane_cx = (np.polyval(lf, y_ev) + np.polyval(rf, y_ev)) / 2.0
    off = car_cx - lane_cx
    if abs(off) > w * DRIFT_THRESHOLD_FRAC:
        return True, ('RIGHT' if off > 0 else 'LEFT'), float(off)
    return False, 'CENTER', float(off)


def draw_lane_overlay(frame, lf, rf, drift, Minv, is_night):
    h, w = frame.shape[:2]
    canvas = np.zeros_like(frame)
    y_vals = np.linspace(0, h - 1, h)

    if lf is not None and rf is not None:
        lx = np.clip(np.polyval(lf, y_vals), 0, w - 1)
        rx = np.clip(np.polyval(rf, y_vals), 0, w - 1)
        pl = np.column_stack((lx, y_vals)).astype(np.int32)
        pr = np.column_stack((rx, y_vals)).astype(np.int32)
        cv2.fillPoly(canvas, [np.vstack([pl, pr[::-1]])], (0, 200, 60))
        cv2.polylines(canvas, [pl], False, (255, 200, 0), 30, cv2.LINE_AA)
        cv2.polylines(canvas, [pr], False, (0, 100, 255), 30, cv2.LINE_AA)

    unwarped = cv2.warpPerspective(canvas, Minv, (w, h))
    result = cv2.addWeighted(frame, 1.0, unwarped, 0.4, 0)

    is_d, direction, off = drift
    hud_col = (0, 50, 220) if is_d else (0, 200, 60)
    label = f'DRIFT {direction}!' if is_d else 'On Lane'
    cv2.putText(result, f'{label} ({off:+.0f}px)', (int(w * 0.35), 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, hud_col, 3, cv2.LINE_AA)

    mode_str = 'NIGHT' if is_night else 'DAY'
    cv2.putText(result, f'Lane Detection | BEV | {mode_str} MODE', (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return result


def process_frame(frame, smoother, mode):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    night = is_night_frame(gray, mode)
    cfg = get_cfg(night)

    filtered, color_mask = adaptive_preprocess(frame, cfg)
    blurred = gaussian_blur(filtered)
    edges = adaptive_canny(blurred, cfg['canny_sigma'])
    combined = cv2.bitwise_and(edges, color_mask)
    combined_roi = apply_roi_mask(combined, frame.shape)

    M, Minv = get_perspective_matrices(frame.shape)
    warped = cv2.warpPerspective(combined_roi, M, (frame.shape[1], frame.shape[0]),
                                 flags=cv2.INTER_LINEAR)

    lpts, rpts = sliding_window_cluster(warped, cfg['window_margin'])
    lf, rf = smoother.update(fit_polynomial(lpts), fit_polynomial(rpts), cfg['ema_alpha'])
    drift = detect_drift(warped.shape, lf, rf)
    return draw_lane_overlay(frame, lf, rf, drift, Minv, night)


def process_video(input_path, output_path, mode='auto', duration=10, progress_cb=None):
    """Processes up to `duration` seconds of `input_path` and writes the
    overlaid result to `output_path` as mp4v (re-encoded to H.264 by the caller
    for browser playback). Mirrors the notebook's Cell 8 `process_video`.

    `progress_cb(percent: int)` is called after each frame, if provided.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open input video: {input_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = max(1, int(duration * fps))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))
    smoother = LaneSmoother()

    try:
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(process_frame(frame, smoother, mode))
            if progress_cb:
                progress_cb(min(99, round((i + 1) / total * 100)))
    finally:
        cap.release()
        out.release()

    if progress_cb:
        progress_cb(100)

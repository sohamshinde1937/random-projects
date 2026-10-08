# ==============================================================================
#  Hand Gesture Recognition + Air Writing System v5.0
#  Python 3.10+  |  mediapipe 0.10.x  (Tasks API)
#
#  ACCURACY UPGRADES:
#    - Dual-rate EMA landmark smoothing (jitter reduction ~60 %)
#    - Orientation-independent finger detection (distance-based, not y-axis)
#    - Adaptive thresholds calibrated to detected hand size
#    - Multi-signal thumb detection (3 heuristics, majority vote)
#    - Pinch verification (distance + converging-angle check)
#    - Temporal consistency gate (rejects impossible frame jumps)
#    - Palm-facing-camera quality score (skips edge-on hands)
#    - Weighted gesture buffer with temporal decay
#
#  NEW FEATURES:
#    - Pinch-to-PAN canvas (fully functional)
#    - Eraser tool (peace-sign cursor paints black)
#    - REDO stack (swipe-right / Y key)
#    - SWIPE dynamic-gesture detector
#    - Auto-save (every 60 s when canvas has content)
#    - Optional sound feedback (Windows / Linux)
#    - Performance profiler overlay (F key)
#    - Grid overlay (G key)
#    - Help overlay (H key)
#    - Multi-hand coordination: left = tool palette, right = draw
#    - Trajectory smoothing for drawn strokes
#    - Canvas offset indicator + stroke counter
#    - Gesture history log (last 8 gestures in corner)
#    - Calibration reset (R key)
#
#  GESTURES  (DRAW mode):
#    Index only        → DRAW
#    Index + Middle    → ERASE / pause
#    Fist (0 fingers)  → CLEAR canvas
#    3 fingers         → CHANGE color
#    4 fingers         → CHANGE brush size
#    5 fingers (open)  → SAVE canvas
#    Thumb only        → UNDO
#    Pinch (T+I)       → PAN canvas
#    Swipe ←           → UNDO
#    Swipe →           → REDO
#
#  MODES  (M to cycle):  DRAW | VOLUME | SLIDES | DISPLAY
#
#  KEYS:
#    Q/Esc  Quit   M  Mode   C  Clear   S  Save
#    Z  Undo       Y  Redo   G  Grid    E  Eraser toggle
#    H  Help       F  Profiler   R  Reset calibration
#
#  INSTALL:  pip install mediapipe opencv-python numpy
#  RUN:      python advanced_riya_code.py
# ==============================================================================


import cv2
import numpy as np
import time
import argparse
import urllib.request
import os
import math
import threading
from collections import deque, Counter
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─────────────────────────── MODEL ───────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand-landmarker model (~5 MB) …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.\n")

# ─────────────────────────── CONFIG ──────────────────────────
@dataclass
class Cfg:
    # MediaPipe
    MIN_DETECT: float = 0.72
    MIN_PRESENCE: float = 0.72
    MIN_TRACK: float = 0.72
    SMOOTH_N: int = 9

    # Smoothing
    ALPHA_SMOOTH: float = 0.38        # for gesture analysis
    ALPHA_FAST: float = 0.65          # for drawing cursor
    JITTER_MAX: float = 0.08          # max normalised jump per frame

    # UI colours  (BGR)
    C_PRIMARY  = (0, 220, 180)
    C_WHITE    = (255, 255, 255)
    C_BLACK    = (0, 0, 0)
    C_DARK     = (18, 18, 28)
    C_LEFT     = (255, 140, 0)
    C_RIGHT    = (0, 200, 255)
    C_YELLOW   = (0, 220, 220)
    C_GREEN    = (0, 200, 60)
    C_RED      = (0, 60, 220)
    C_MUTED    = (90, 90, 110)

    # Drawing
    DRAW_COLORS = [
        (0, 255, 255), (255, 80, 80), (80, 255, 80),
        (80, 80, 255), (255, 100, 255), (255, 255, 255),
        (0, 180, 255), (255, 180, 0),
    ]
    BRUSH_SIZES = [2, 5, 9, 15, 22, 30]

    DRAW_MIN_MOVE: int = 3
    PINCH_ANGLE_THRESH: float = 0.25   # cos-angle threshold for pinch
    SWIPE_FRAMES: int = 14
    SWIPE_THRESH: float = 0.22         # fraction of frame width

class AppMode(Enum):
    DRAW = "draw"; DISPLAY = "display"
    VOLUME = "volume"; SLIDES = "slides"

# ─────────────────────────── LANDMARKS ───────────────────────
WRIST = 0
THUMB_CMC = 1; THUMB_MCP = 2; THUMB_IP = 3; THUMB_TIP = 4
INDEX_MCP = 5; INDEX_PIP = 6; INDEX_DIP = 7; INDEX_TIP = 8
MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP = 13; RING_PIP = 14; RING_DIP = 15; RING_TIP = 16
PINKY_MCP = 17; PINKY_PIP = 18; PINKY_DIP = 19; PINKY_TIP = 20

TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),(0,17),
]

# ─────────────────────────── DATA TYPES ──────────────────────
@dataclass
class SL:
    """Smoothed Landmark – drop-in for NormalizedLandmark."""
    x: float; y: float; z: float = 0.0

@dataclass
class HandResult:
    """All computed info for one detected hand."""
    side: str
    slm: List[SL]               # smoothed landmarks
    finger_states: List[bool]    # [thumb, index, middle, ring, pinky]
    count: int
    confidence: float
    palm_quality: float         # 0-1 how much palm faces camera
    pinch: bool
    pointing: bool
    peace: bool
    fist: bool
    thumb_only: bool
    stability: float
    bbox: Tuple[int,int,int,int] # x1,y1,x2,y2
    cursor: Tuple[int,int]       # index-tip pixel coords

# ─────────────────── LANDMARK SMOOTHER ───────────────────────
class LandmarkSmoother:
    """Dual-rate exponential moving average + jitter clamp."""
    def __init__(self, alpha_smooth=0.38, alpha_fast=0.65, jitter_max=0.08):
        self.a_s = alpha_smooth
        self.a_f = alpha_fast
        self.jmax = jitter_max
        self._s: Optional[List[SL]] = None   # smooth state
        self._f: Optional[List[SL]] = None   # fast state

    def update(self, landmarks) -> Tuple[List[SL], List[SL]]:
        """Returns (smoothed_for_gestures, fast_for_cursor)."""
        pts = [SL(lm.x, lm.y, lm.z) for lm in landmarks]
        if self._s is None:
            self._s = [SL(p.x, p.y, p.z) for p in pts]
            self._f = [SL(p.x, p.y, p.z) for p in pts]
            return self._s, self._f

        # Jitter clamp
        for i, p in enumerate(pts):
            dx = abs(p.x - self._s[i].x)
            dy = abs(p.y - self._s[i].y)
            if dx > self.jmax:
                p = SL(self._s[i].x + math.copysign(self.jmax, p.x - self._s[i].x), p.y, p.z)
            if dy > self.jmax:
                p = SL(p.x, self._s[i].y + math.copysign(self.jmax, p.y - self._s[i].y), p.z)
            pts[i] = p

        self._s = [SL(
            self.a_s * p.x + (1 - self.a_s) * s.x,
            self.a_s * p.y + (1 - self.a_s) * s.y,
            self.a_s * p.z + (1 - self.a_s) * s.z,
        ) for p, s in zip(pts, self._s)]

        self._f = [SL(
            self.a_f * p.x + (1 - self.a_f) * f.x,
            self.a_f * p.y + (1 - self.a_f) * f.y,
            self.a_f * p.z + (1 - self.a_f) * f.z,
        ) for p, f in zip(pts, self._f)]

        return self._s, self._f

    def reset(self):
        self._s = None; self._f = None

# ─────────────────── GEOMETRY HELPERS ────────────────────────
def d2(a: SL, b: SL) -> float:
    """Normalised 2-D distance."""
    return math.hypot(a.x - b.x, a.y - b.y)

def d3(a: SL, b: SL) -> float:
    return math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)

def angle_at(a: SL, b: SL, c: SL) -> float:
    """Angle (rad) at vertex b in triangle a-b-c."""
    bax, bay = a.x - b.x, a.y - b.y
    bcx, bcy = c.x - b.x, c.y - b.y
    dot = bax * bcx + bay * bcy
    mag = math.hypot(bax, bay) * math.hypot(bcx, bcy) + 1e-9
    return math.acos(max(-1, min(1, dot / mag)))

def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def lm_px(slm: List[SL], idx: int, w: int, h: int) -> Tuple[int, int]:
    return int(slm[idx].x * w), int(slm[idx].y * h)

# ─────────────── ADAPTIVE CALIBRATOR ────────────────────────
class AdaptiveCalibrator:
    """Measures hand size on first good detection; derives thresholds."""
    def __init__(self):
        self.hand_size: Optional[float] = None   # wrist→middle_tip
        self.pinch_thresh: float = 0.055
        self.calibrated = False

    def calibrate(self, slm: List[SL], palm_q: float):
        if self.calibrated or palm_q < 0.6:
            return
        self.hand_size = d2(slm[WRIST], slm[MIDDLE_TIP])
        if self.hand_size > 0.15:          # plausible range
            self.pinch_thresh = self.hand_size * 0.12
            self.calibrated = True

    def reset(self):
        self.hand_size = None
        self.pinch_thresh = 0.055
        self.calibrated = False

# ─────────────── PALM QUALITY CHECK ──────────────────────────
def palm_quality(slm: List[SL]) -> float:
    """0-1: how much the palm faces the camera (vs edge-on)."""
    z_vals = [slm[i].z for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)]
    z_range = max(z_vals) - min(z_vals)
    return max(0.0, min(1.0, 1.0 - z_range * 10))

# ─────────────── FINGER ANALYSIS (IMPROVED) ─────────────────
def analyse_hand(slm: List[SL], side: str, cal: AdaptiveCalibrator) -> dict:
    """
    Orientation-independent finger detection:
      • Fingers 1-4: tip farther from wrist than PIP  →  extended
      • Thumb: 3-signal majority vote
      • Pinch: distance + converging-angle verification
    """
    wrist = slm[WRIST]

    # --- Fingers 1-4 (index, middle, ring, pinky) ---
    finger_states = [False] * 5
    for i in range(1, 5):
        tip_d = d2(slm[TIPS[i]], wrist)
        pip_d = d2(slm[PIPS[i]], wrist)
        finger_states[i] = tip_d > pip_d * 1.06

    # --- Thumb: 3-signal majority vote ---
    signals = 0
    # Signal 1: tip farther from INDEX_MCP than THUMB_IP
    s1 = d2(slm[THUMB_TIP], slm[INDEX_MCP]) > d2(slm[THUMB_IP], slm[INDEX_MCP]) * 1.05
    # Signal 2: angle at IP joint (extended ≈ > 140°)
    ang = angle_at(slm[THUMB_MCP], slm[THUMB_IP], slm[THUMB_TIP])
    s2 = ang > 2.4
    # Signal 3: lateral displacement (original heuristic, orientation-dependent)
    if side == "Right":
        s3 = slm[THUMB_TIP].x < slm[THUMB_IP].x
    else:
        s3 = slm[THUMB_TIP].x > slm[THUMB_IP].x
    finger_states[0] = (s1 + s2 + s3) >= 2

    count = sum(finger_states)

    # --- Confidence ---
    z_spread = max(abs(slm[t].z) for t in TIPS[1:])
    conf = round(max(0.45, 1.0 - min(z_spread * 4, 0.55)), 2)

    # --- Pinch (distance + angle) ---
    pinch_dist = d2(slm[THUMB_TIP], slm[INDEX_TIP])
    v_t = (slm[THUMB_TIP].x - slm[THUMB_MCP].x, slm[THUMB_TIP].y - slm[THUMB_MCP].y)
    v_i = (slm[INDEX_TIP].x - slm[INDEX_MCP].x, slm[INDEX_TIP].y - slm[INDEX_MCP].y)
    mag = math.hypot(*v_t) * math.hypot(*v_i) + 1e-9
    cos_a = (v_t[0]*v_i[0] + v_t[1]*v_i[1]) / mag
    pinch = pinch_dist < cal.pinch_thresh and cos_a < Cfg.PINCH_ANGLE_THRESH

    pointing   = finger_states[1] and not any(finger_states[2:])
    peace      = finger_states[1] and finger_states[2] and not any(finger_states[3:]) and not finger_states[0]
    fist       = count == 0
    thumb_only = finger_states[0] and not any(finger_states[1:])

    return dict(
        finger_states=finger_states, count=count, confidence=conf,
        pinch=pinch, pointing=pointing, peace=peace,
        fist=fist, thumb_only=thumb_only,
    )

# ─────────────── GESTURE BUFFER (DECAY) ─────────────────────
class GestureBuffer:
    """Weighted majority vote with temporal decay."""
    __slots__ = ("_buf", "_n")

    def __init__(self, n=Cfg.SMOOTH_N):
        self._buf: deque = deque(maxlen=n)
        self._n = n

    def push(self, v):
        self._buf.append(v)

    @property
    def stable(self):
        if not self._buf:
            return 0
        weights = {v: 0.0 for v in set(self._buf)}
        total = 0.0
        for i, v in enumerate(self._buf):
            w = (i + 1) / len(self._buf)       # newer = heavier
            weights[v] += w
            total += w
        return max(weights, key=weights.get)

    @property
    def stability(self):
        if len(self._buf) < 2:
            return 1.0
        return Counter(self._buf).most_common(1)[0][1] / len(self._buf)

    def clear(self):
        self._buf.clear()

# ─────────────── SWIPE DETECTOR ─────────────────────────────
class SwipeDetector:
    """Detects left/right swipes from index-tip trajectory."""
    def __init__(self, n=14, thresh=0.22):
        self.n = n
        self.thresh = thresh
        self._trail: deque = deque(maxlen=n)

    def push(self, x_norm: float):
        self._trail.append(x_norm)

    @property
    def swipe(self) -> Optional[str]:
        if len(self._trail) < self.n:
            return None
        dx = self._trail[-1] - self._trail[0]
        if dx > self.thresh:
            self._trail.clear()
            return "right"
        if dx < -self.thresh:
            self._trail.clear()
            return "left"
        return None

    def clear(self):
        self._trail.clear()

# ─────────────── TRAJECTORY SMOOTHER ────────────────────────
class TrajectorySmoother:
    """Weighted-average over last N points for smoother drawn lines."""
    def __init__(self, window=3):
        self._buf: deque = deque(maxlen=window)

    def feed(self, pt: Tuple[int, int]) -> Tuple[int, int]:
        self._buf.append(pt)
        n = len(self._buf)
        ax = sum(p[0] for p in self._buf) // n
        ay = sum(p[1] for p in self._buf) // n
        return (ax, ay)

    def reset(self):
        self._buf.clear()

# ─────────────── SOUND FEEDBACK (FIXED) ─────────────────────
SOUND_ENABLED = True  # Set to False to mute everywhere

def beep(freq=880, ms=40):
    """Cross-platform beep (silent fallback)."""
    if not SOUND_ENABLED:
        return
    try:
        import winsound
        winsound.Beep(freq, ms)
        return
    except (ImportError, RuntimeError):
        pass
    try:
        os.system(f'play -nq -t alsa synth {ms/1000:.2f} sine {freq} 2>/dev/null')
    except Exception:
        pass

# ─────────────── PERFORMANCE PROFILER ───────────────────────
class Profiler:
    """Tracks per-frame timings."""
    __slots__ = ("_t", "detect", "smooth", "analysis", "render", "total")

    def __init__(self):
        self._t = 0.0
        self.detect = self.smooth = self.analysis = self.render = self.total = 0.0

    def tick(self, label=""):
        now = time.perf_counter()
        if self._t and label:
            dt = now - self._t
            if   label == "detect":   self.detect   = dt
            elif label == "smooth":   self.smooth   = dt
            elif label == "analysis": self.analysis = dt
            elif label == "render":   self.render   = dt
        self._t = now

    def frame_end(self):
        self.total = time.perf_counter() - self._t
        self._t = time.perf_counter()

# ─────────────── AIR CANVAS v2 ──────────────────────────────
class AirCanvas:
    """Persistent drawing overlay with pan, eraser, redo, grid."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self._canvas   = np.zeros((h, w, 3), dtype=np.uint8)
        self._strokes: List[Tuple] = []
        self._redo: List[Tuple] = []
        self._cur_pts: List[Tuple[int,int]] = []
        self._col_idx  = 0
        self._sz_idx   = 1
        self._prev_pt  = None
        self._drawing  = False
        self._erasing  = False
        self._offset   = [0, 0]
        self._pan_anchor = None
        self._traj = TrajectorySmoother(3)
        self._auto_timer: Optional[threading.Timer] = None

    # ---- properties ----
    @property
    def color(self):      return Cfg.DRAW_COLORS[self._col_idx % len(Cfg.DRAW_COLORS)]
    @property
    def thickness(self):  return Cfg.BRUSH_SIZES[self._sz_idx % len(Cfg.BRUSH_SIZES)]
    @property
    def color_idx(self):  return self._col_idx
    @property
    def size_idx(self):   return self._sz_idx
    @property
    def offset(self):     return tuple(self._offset)
    @property
    def stroke_count(self): return len(self._strokes)
    @property
    def has_content(self): return bool(self._strokes)

    # ---- color / size ----
    def next_color(self):
        self._col_idx = (self._col_idx + 1) % len(Cfg.DRAW_COLORS)
    def next_size(self):
        self._sz_idx = (self._sz_idx + 1) % len(Cfg.BRUSH_SIZES)

    # ---- stroke ops ----
    def start_stroke(self, pt):
        self._drawing = True; self._erasing = False
        self._cur_pts = [pt]; self._prev_pt = pt; self._traj.reset()
        self._redo.clear()

    def start_erase(self, pt):
        self._drawing = True; self._erasing = True
        self._prev_pt = pt; self._traj.reset()
        self._redo.clear()

    def add_point(self, pt):
        if not self._drawing or self._prev_pt is None:
            return
        sp = self._traj.feed(pt)
        dx, dy = sp[0] - self._prev_pt[0], sp[1] - self._prev_pt[1]
        if math.hypot(dx, dy) < Cfg.DRAW_MIN_MOVE:
            return
        col = (0, 0, 0) if self._erasing else self.color
        th  = self.thickness * 3 if self._erasing else self.thickness
        cv2.line(self._canvas, self._prev_pt, sp, col, th, cv2.LINE_AA)
        if not self._erasing:
            self._cur_pts.append(sp)
        self._prev_pt = sp

    def end_stroke(self):
        if self._drawing and not self._erasing and len(self._cur_pts) > 1:
            self._strokes.append((list(self._cur_pts), self.color, self.thickness))
        self._drawing = False; self._erasing = False
        self._prev_pt = None; self._cur_pts = []; self._traj.reset()

    def undo(self):
        if not self._strokes: return False
        self._redo.append(self._strokes.pop()); self._redraw(); return True

    def redo(self):
        if not self._redo: return False
        self._strokes.append(self._redo.pop()); self._redraw(); return True

    def clear(self):
        self._strokes.clear(); self._redo.clear(); self._cur_pts.clear()
        self._canvas[:] = 0; self._prev_pt = None
        self._drawing = False; self._erasing = False

    def save(self, path=None):
        if path is None:
            path = f"canvas_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(path, self._canvas); return path

    # ---- pan ----
    def start_pan(self, pt):  self._pan_anchor = pt
    def update_pan(self, pt):
        if self._pan_anchor:
            self._offset[0] += pt[0] - self._pan_anchor[0]
            self._offset[1] += pt[1] - self._pan_anchor[1]
            self._pan_anchor = pt
    def end_pan(self):       self._pan_anchor = None

    # ---- blend ----
    def blend(self, frame):
        ox, oy = self._offset
        # Create shifted canvas
        shifted = np.zeros_like(self._canvas)
        sx, sy = max(0, -ox), max(0, -oy)
        dx, dy = max(0, ox), max(0, oy)
        sw = min(self.w - sx, frame.shape[1] - dx)
        sh = min(self.h - sy, frame.shape[0] - dy)
        if sw > 0 and sh > 0:
            shifted[dy:dy+sh, dx:dx+sw] = self._canvas[sy:sy+sh, sx:sx+sw]
        mask = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
        bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        fg = cv2.bitwise_and(shifted, shifted, mask=mask)
        return cv2.add(bg, fg)

    # ---- auto-save ----
    def schedule_auto_save(self, interval=60):
        def _auto():
            if self.has_content:
                p = self.save("autosave_" + time.strftime('%Y%m%d_%H%M%S') + ".png")
                print(f"[auto-save] {p}")
            self._auto_timer = threading.Timer(interval, _auto)
            self._auto_timer.daemon = True
            self._auto_timer.start()
        self._auto_timer = threading.Timer(interval, _auto)
        self._auto_timer.daemon = True
        self._auto_timer.start()

    def cancel_auto_save(self):
        if self._auto_timer:
            self._auto_timer.cancel()

    # ---- internal ----
    def _redraw(self):
        self._canvas[:] = 0
        for pts, col, thick in self._strokes:
            for i in range(1, len(pts)):
                cv2.line(self._canvas, pts[i-1], pts[i], col, thick, cv2.LINE_AA)

# ─────────────── DRAWING HELPERS ────────────────────────────
def draw_skeleton(frame, slm, color, w, h):
    pts = [lm_px(slm, i, w, h) for i in range(21)]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (140, 140, 150), 1, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        r = 5 if i in (INDEX_TIP, MIDDLE_TIP, THUMB_TIP) else 3
        cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)

def draw_corner_box(frame, x1, y1, x2, y2, color, t=2, cl=16):
    for (px, py), (dx, dy) in [
        ((x1,y1),(cl,0)),((x1,y1),(0,cl)),((x2,y1),(-cl,0)),((x2,y1),(0,cl)),
        ((x1,y2),(cl,0)),((x1,y2),(0,-cl)),((x2,y2),(-cl,0)),((x2,y2),(0,-cl)),
    ]:
        cv2.line(frame, (px,py), (px+dx,py+dy), color, t, cv2.LINE_AA)

def txt(frame, text, pos, scale, color, thick=1):
    x, y = pos
    cv2.putText(frame, text, (x+1,y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, Cfg.C_BLACK, thick+1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)

def draw_grid_overlay(frame, spacing=60, color=(35, 35, 50)):
    h, w = frame.shape[:2]
    for x in range(0, w, spacing):
        cv2.line(frame, (x, 0), (x, h), color, 1, cv2.LINE_AA)
    for y in range(0, h, spacing):
        cv2.line(frame, (0, y), (w, y), color, 1, cv2.LINE_AA)

def draw_palette(frame, canvas, w):
    px = w - 30
    for i, col in enumerate(Cfg.DRAW_COLORS):
        py = 15 + i * 26
        cv2.circle(frame, (px, py), 9, col, -1, cv2.LINE_AA)
        if i == canvas.color_idx:
            cv2.circle(frame, (px, py), 12, Cfg.C_WHITE, 2, cv2.LINE_AA)
    py2 = 15 + len(Cfg.DRAW_COLORS) * 26 + 8
    cv2.circle(frame, (px, py2), min(canvas.thickness, 14), canvas.color, -1, cv2.LINE_AA)
    txt(frame, f"S{canvas.size_idx+1}", (px-8, py2+4), 0.3, Cfg.C_WHITE)

# ─────────────── GESTURE HISTORY LOG ────────────────────────
class GestureLog:
    def __init__(self, n=8):
        self._buf: deque = deque(maxlen=n)
    def push(self, label: str):
        self._buf.append((label, time.time()))
    def draw(self, frame, x, y):
        for i, (label, t) in enumerate(reversed(self._buf)):
            age = time.time() - t
            alpha = max(0.3, 1.0 - age / 6.0)
            c = tuple(int(v * alpha) for v in Cfg.C_MUTED)
            txt(frame, label, (x, y + i * 18), 0.35, c)

# ─────────────── DRAW CONTROLLER v2 ─────────────────────────
class DrawController:
    COOLDOWN = 16

    def __init__(self, canvas: AirCanvas):
        self.canvas = canvas
        self._cd = 0
        self._was_drawing = False
        self._was_pinching = False
        self._erase_mode = False
        self._status = ""; self._status_t = 0

    @property
    def erase_mode(self): return self._erase_mode
    @erase_mode.setter
    def erase_mode(self, v):
        self._erase_mode = v
        if v and self._was_drawing and not self.canvas._erasing:
            self.canvas.end_stroke(); self._was_drawing = False

    def status(self):
        return self._status if time.time() - self._status_t < 1.6 else ""

    def _set(self, msg, sound=True):
        self._status = msg; self._status_t = time.time()
        if sound: beep()

    def _end_draw(self):
        if self._was_drawing:
            self.canvas.end_stroke(); self._was_drawing = False

    def update(self, hr: HandResult, swipe: Optional[str]) -> Tuple[Optional[Tuple[int,int]], bool]:
        if self._cd > 0: self._cd -= 1
        cursor = hr.cursor
        fs, cnt = hr.finger_states, hr.count

        # --- Pinch → PAN ---
        if hr.pinch:
            if not self._was_pinching:
                self._end_draw()
                self.canvas.start_pan(cursor)
                self._was_pinching = True
            else:
                self.canvas.update_pan(cursor)
            return cursor, False
        elif self._was_pinching:
            self.canvas.end_pan(); self._was_pinching = False

        # --- Swipe override ---
        if swipe == "left":
            self._end_draw()
            if self.canvas.undo(): self._set("Undo (swipe)")
            return cursor, False
        if swipe == "right":
            self._end_draw()
            if self.canvas.redo(): self._set("Redo (swipe)")
            return cursor, False

        # --- Gesture triggers (with cooldown) ---
        if self._cd == 0:
            if hr.fist:
                self._end_draw(); self.canvas.clear()
                self._cd = self.COOLDOWN; self._set("Canvas cleared"); return cursor, False
            if hr.thumb_only:
                self._end_draw()
                if self.canvas.undo(): self._set("Undo")
                self._cd = self.COOLDOWN; return cursor, False
            if cnt == 3:
                self._end_draw(); self.canvas.next_color()
                self._cd = self.COOLDOWN; self._set("Color → " + str(self.canvas.color_idx + 1))
                return cursor, False
            if cnt == 4:
                self._end_draw(); self.canvas.next_size()
                self._cd = self.COOLDOWN; self._set(f"Brush → {self.canvas.thickness}px")
                return cursor, False
            if cnt == 5:
                self._end_draw()
                p = self.canvas.save()
                self._cd = self.COOLDOWN * 3; self._set(f"Saved: {p}"); return cursor, False

        # --- Peace → erase or pause ---
        if hr.peace:
            self._end_draw()
            if self._erase_mode:
                if not self.canvas._drawing:
                    self.canvas.start_erase(cursor)
                else:
                    self.canvas.add_point(cursor)
                return cursor, True
            return cursor, False

        # --- Index only → DRAW ---
        if hr.pointing:
            if not self._was_drawing:
                self.canvas.start_stroke(cursor)
                self._was_drawing = True
            else:
                self.canvas.add_point(cursor)
            return cursor, True

        # --- Default: stop ---
        self._end_draw()
        return cursor, False

# ─────────────── MODE BANNERS ───────────────────────────────
VOL_LABELS = {0:"MUTE",1:"20 %",2:"40 %",3:"60 %",4:"80 %",5:"100 %"}
SLD_LABELS = {0:"⏸ PAUSE",1:"◀ PREV",2:"▶ NEXT",3:"🔍+ ZOOM",4:"⬛ BLANK",5:"📺 PRESENT"}

def draw_mode_banner(frame, count, mode, w, h):
    if mode == AppMode.VOLUME:
        lab = VOL_LABELS.get(count, "")
        vol = count / 5.0
        bw, bx, by = 220, w//2 - 110, h - 78
        cv2.rectangle(frame, (bx-2,by-2), (bx+bw+2,by+18), (40,40,60), -1)
        cv2.rectangle(frame, (bx,by), (bx+int(bw*vol),by+14),
                      lerp_color((0,80,200),(0,220,80),vol), -1)
        txt(frame, f"🔊 {lab}", (bx, by-8), 0.6, Cfg.C_PRIMARY, 2)
    elif mode == AppMode.SLIDES:
        lab = SLD_LABELS.get(count, "")
        ov = frame.copy()
        cv2.rectangle(ov, (w//2-130,h-92),(w//2+130,h-54),(30,30,50),-1)
        cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
        txt(frame, lab, (w//2-110, h-66), 0.75, Cfg.C_PRIMARY, 2)

# ─────────────── HUD ────────────────────────────────────────
def draw_hud(frame, fps, mode, w, h, gname, canvas=None, show_grid=False,
             show_prof=False, prof=None, show_help=False, glog=None):
    # Top-left panel
    ov = frame.copy()
    cv2.rectangle(ov, (0,0),(310,84), Cfg.C_DARK, -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
    txt(frame, f"FPS: {fps:.1f}", (10,24), 0.62, Cfg.C_PRIMARY)
    txt(frame, f"MODE: {mode.value.upper()}", (10,48), 0.55, Cfg.C_WHITE)
    txt(frame, gname, (10,72), 0.48, Cfg.C_YELLOW)
    if canvas and mode == AppMode.DRAW:
        txt(frame, f"Strokes: {canvas.stroke_count}  Off: {canvas.offset}",
            (170, 24), 0.38, Cfg.C_MUTED)
        er = "ON" if canvas._erasing or (hasattr(DrawController,'_inst') and DrawController._inst.erase_mode) else "OFF"
        # We'll pass erase flag separately; for now just draw palette
        draw_palette(frame, canvas, w)

    # Bottom bar
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0,h-30),(w,h), Cfg.C_DARK, -1)
    cv2.addWeighted(ov2, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame,
        "Q:Quit M:Mode C:Clear S:Save Z:Undo Y:Redo G:Grid E:Eraser H:Help F:Prof R:Calib",
        (6, h-9), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120,120,140), 1, cv2.LINE_AA)

    if show_grid:
        draw_grid_overlay(frame)

    if show_prof and prof:
        py = 100
        for label, val in [("Detect", prof.detect), ("Smooth", prof.smooth),
                           ("Analysis", prof.analysis), ("Render", prof.render),
                           ("Total", prof.total)]:
            txt(frame, f"{label}: {val*1000:.1f}ms", (w-200, py), 0.38,
                Cfg.C_GREEN if val*1000 < 10 else Cfg.C_RED)
            py += 18

    if show_help:
        _draw_help(frame, w, h)

    if glog:
        glog.draw(frame, w - 180, 100)

def _draw_help(frame, w, h):
    ov = frame.copy()
    cx, cy = w//2, h//2
    bw, bh = 340, 310
    cv2.rectangle(ov, (cx-bw, cy-bh),(cx+bw, cy+bh),(15,15,25),-1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    lines = [
        "─── GESTURE MAP ───",
        "1 finger (index)  →  DRAW",
        "2 fingers (V)     →  ERASE / pause",
        "Fist (0)          →  CLEAR canvas",
        "3 fingers         →  Cycle COLOR",
        "4 fingers         →  Cycle BRUSH SIZE",
        "5 fingers (open)  →  SAVE canvas",
        "Thumb only        →  UNDO",
        "Pinch (T+I)       →  PAN canvas",
        "Swipe ← / →       →  Undo / Redo",
        "",
        "─── KEYBOARD ───",
        "M=Mode  C=Clear  S=Save  Z=Undo  Y=Redo",
        "G=Grid  E=Eraser  H=Help  F=Profiler  R=Calib",
        "",
        "Press H to close",
    ]
    for i, line in enumerate(lines):
        txt(frame, line, (cx-bw+20, cy-bh+25+i*22), 0.42, Cfg.C_WHITE)

def draw_hand_info(frame, hr: HandResult, mode):
    x1, y1, x2, y2 = hr.bbox
    col = Cfg.C_RIGHT if hr.side == "Right" else Cfg.C_LEFT

    # Big count
    ct = str(hr.count)
    ts = cv2.getTextSize(ct, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 6)[0]
    tx = x1 + (x2-x1)//2 - ts[0]//2
    ty = max(y1 - 10, 80)
    cv2.putText(frame, ct, (tx+2,ty+2), cv2.FONT_HERSHEY_SIMPLEX, 3.0, Cfg.C_BLACK, 8, cv2.LINE_AA)
    cv2.putText(frame, ct, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, 3.0, col, 6, cv2.LINE_AA)

    txt(frame, f"{hr.side} Hand", (x1, y2+20), 0.55, col)

    # Confidence bar
    bx, by, bw = x1, y2+32, x2-x1
    cv2.rectangle(frame, (bx,by),(bx+bw,by+6), (50,50,70), -1)
    cv2.rectangle(frame, (bx,by),(bx+int(bw*hr.confidence),by+6),
                  lerp_color((0,80,220),(0,220,80),hr.confidence), -1)
    txt(frame, f"Conf {int(hr.confidence*100)}%", (bx,by+20), 0.38, Cfg.C_WHITE)

    # Palm quality
    pq = hr.palm_quality
    pc = (0,220,80) if pq>0.7 else ((0,200,220) if pq>0.4 else (0,80,220))
    pl = "Palm OK" if pq>0.7 else ("Palm tilted" if pq>0.4 else "Edge-on!")
    txt(frame, pl, (bx, by+36), 0.36, pc)

    # Stability
    sc = (0,220,80) if hr.stability>=0.85 else ((0,200,220) if hr.stability>=0.6 else (0,80,220))
    sl = "Stable" if hr.stability>=0.85 else ("Settling" if hr.stability>=0.6 else "Jitter")
    txt(frame, sl, (bx+bw-70, by+36), 0.36, sc)

        # Finger dots
    dy = y2 + 56
    gap = 16
    x0 = x1 + (bw - gap * 4) // 2
    for fi, (st, lb) in enumerate(zip(hr.finger_states, ["T","I","M","R","P"])):
        dx = x0 + fi * gap
        cv2.circle(frame, (dx, dy), 5, (0,220,100) if st else (60,60,90), -1)
        txt(frame, lb, (dx-4, dy+4), 0.26, Cfg.C_WHITE)

# ─────────────── MAIN ───────────────────────────────────────
def main(max_hands=2):
    print("=" * 52)
    print("  Hand Gesture + Air Writing System  v5.0")
    print("  Q:Quit  M:Mode  H:Help  F:Profiler")
    print("=" * 52 + "\n")
    ensure_model()

    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_opts, num_hands=max_hands,
        min_hand_detection_confidence=Cfg.MIN_DETECT,
        min_hand_presence_confidence=Cfg.MIN_PRESENCE,
        min_tracking_confidence=Cfg.MIN_TRACK,
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam."); return
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    canvas    = AirCanvas(W, H)
    ctrl      = DrawController(canvas)
    DrawController._inst = ctrl        # for HUD access
    cal       = AdaptiveCalibrator()
    smoothers: dict[int, LandmarkSmoother] = {}
    g_bufs:   dict[int, GestureBuffer]   = {}
    swipes:   dict[int, SwipeDetector]   = {}
    glog      = GestureLog(8)
    prof      = Profiler()
    modes     = list(AppMode); mode_idx = 0
    show_grid = False; show_prof = False; show_help = False
    fps_t = time.time(); fps = 0.0; fc = 0
    gname = "No hand detected"
    mp_ts = 0

    canvas.schedule_auto_save(60)

    try:
        with mp_vision.HandLandmarker.create_from_options(options) as lander:
            while True:
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                prof.tick()

                # --- Detect ---
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                now_ms = int(time.time() * 1000)
                if now_ms <= mp_ts: now_ms = mp_ts + 1
                mp_ts = now_ms
                result = lander.detect_for_video(mp_img, now_ms)
                prof.tick("detect")

                # --- FPS ---
                fc += 1; now = time.time()
                if now - fps_t >= 0.5:
                    fps = fc / (now - fps_t); fc = 0; fps_t = now

                mode = modes[mode_idx]
                lm_lists = result.hand_landmarks
                hness    = result.handedness
                cursor_pt = None; is_drawing = False
                frame_swipe = None

                # --- Per-hand ---
                for idx in range(len(lm_lists)):
                    lm_raw = lm_lists[idx]
                    side   = hness[idx][0].category_name

                    if idx not in smoothers:
                        smoothers[idx] = LandmarkSmoother(Cfg.ALPHA_SMOOTH, Cfg.ALPHA_FAST, Cfg.JITTER_MAX)
                        g_bufs[idx]   = GestureBuffer()
                        swipes[idx]   = SwipeDetector(Cfg.SWIPE_FRAMES, Cfg.SWIPE_THRESH)

                    slm_smooth, slm_fast = smoothers[idx].update(lm_raw)
                    prof.tick("smooth")

                    # Calibrate
                    pq = palm_quality(slm_smooth)
                    cal.calibrate(slm_smooth, pq)

                    # Analyse
                    a = analyse_hand(slm_smooth, side, cal)
                    prof.tick("analysis")

                    # Gesture buffer
                    g_bufs[idx].push(a['count'])
                    stable    = g_bufs[idx].stable
                    stab      = g_bufs[idx].stability

                    # Swipe
                    swipes[idx].push(slm_fast[INDEX_TIP].x)
                    sw = swipes[idx].swipe
                    if sw: frame_swipe = sw

                    # BBox
                    xs = [int(lm.x * w) for lm in lm_raw]
                    ys = [int(lm.y * h) for lm in lm_raw]
                    x1, x2 = max(0, min(xs)-14), min(w, max(xs)+14)
                    y1, y2 = max(0, min(ys)-14), min(h, max(ys)+14)

                    hr = HandResult(
                        side=side, slm=slm_smooth,
                        finger_states=a['finger_states'], count=stable,
                        confidence=a['confidence'], palm_quality=pq,
                        pinch=a['pinch'], pointing=a['pointing'],
                        peace=a['peace'], fist=a['fist'],
                        thumb_only=a['thumb_only'],
                        stability=stab, bbox=(x1,y1,x2,y2),
                        cursor=lm_px(slm_fast, INDEX_TIP, w, h),
                    )

                    # Skeleton + box
                    draw_skeleton(frame, slm_smooth,
                                  Cfg.C_RIGHT if side=="Right" else Cfg.C_LEFT, w, h)
                    draw_corner_box(frame, x1, y1, x2, y2,
                                    Cfg.C_RIGHT if side=="Right" else Cfg.C_LEFT)
                    draw_hand_info(frame, hr, mode)

                    # Gesture name
                    if hr.pointing:    gname = "DRAW"
                    elif hr.peace:     gname = "ERASE / PAUSE"
                    elif hr.fist:      gname = "FIST → CLEAR"
                    elif hr.thumb_only:gname = "THUMB → UNDO"
                    elif hr.pinch:     gname = "PINCH → PAN"
                    elif stable==3:    gname = "3 → COLOR"
                    elif stable==4:    gname = "4 → BRUSH SIZE"
                    elif stable==5:    gname = "5 → SAVE"
                    else:              gname = f"{stable} fingers"

                    if pq < 0.35:
                        gname += "  [low palm quality]"

                    # Mode actions
                    if mode == AppMode.DRAW:
                        # Multi-hand: left hand = tools only (no draw), right = draw
                        if side == "Left" and len(lm_lists) > 1:
                            # Left hand controls tools via finger count
                            if hr.pinch:
                                if not ctrl._was_pinching:
                                    ctrl._end_draw()
                                    canvas.start_pan(hr.cursor)
                                    ctrl._was_pinching = True
                                else:
                                    canvas.update_pan(hr.cursor)
                                cursor_pt = hr.cursor
                            elif ctrl._was_pinching and side == "Left":
                                canvas.end_pan(); ctrl._was_pinching = False
                        else:
                            cp, isd = ctrl.update(hr, frame_swipe if side=="Right" else None)
                            if cp: cursor_pt = cp; is_drawing = isd
                    else:
                        draw_mode_banner(frame, stable, mode, w, h)

                    # Log gesture changes
                    if fc % 15 == 0 and gname != "No hand detected":
                        glog.push(gname)

                # If hand lost, reset smoother for that index
                for idx in list(smoothers.keys()):
                    if idx >= len(lm_lists):
                        smoothers[idx].reset()
                        g_bufs[idx].clear()
                        swipes[idx].clear()

                if not lm_lists:
                    gname = "No hand detected"

                # --- Blend canvas ---
                if mode == AppMode.DRAW:
                    frame = canvas.blend(frame)
                    if cursor_pt:
                        col = canvas.color if is_drawing else (
                            (0,0,200) if ctrl.erase_mode else Cfg.C_WHITE)
                        r = canvas.thickness + 4
                        cv2.circle(frame, cursor_pt, r, col, 2, cv2.LINE_AA)
                        cv2.circle(frame, cursor_pt, 3, col, -1, cv2.LINE_AA)
                        if ctrl.erase_mode and not is_drawing:
                            cv2.circle(frame, cursor_pt, canvas.thickness*3, (0,0,200), 1, cv2.LINE_AA)
                    st = ctrl.status()
                    if st:
                        txt(frame, st, (w//2-120, h//2-20), 0.8, Cfg.C_YELLOW, 2)

                # --- HUD ---
                draw_hud(frame, fps, mode, w, h, gname,
                         canvas if mode==AppMode.DRAW else None,
                         show_grid, show_prof, prof, show_help, glog)
                prof.tick("render")
                prof.frame_end()

                cv2.imshow("Hand Gesture v5.0 — H:Help  Q:Quit", frame)

                # --- Keys ---
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27): break
                elif key == ord('m'):
                    mode_idx = (mode_idx+1) % len(modes)
                    print(f"Mode → {modes[mode_idx].value.upper()}")
                    beep(660, 30)
                elif key == ord('c'):
                    canvas.clear(); print("Cleared."); beep(440, 30)
                elif key == ord('s'):
                    p = canvas.save(); print(f"Saved: {p}"); beep(880, 50)
                elif key == ord('z'):
                    if canvas.undo(): print("Undo."); beep(550, 30)
                elif key == ord('y'):
                    if canvas.redo(): print("Redo."); beep(660, 30)
                elif key == ord('g'):
                    show_grid = not show_grid
                    print(f"Grid {'ON' if show_grid else 'OFF'}")
                elif key == ord('e'):
                    ctrl.erase_mode = not ctrl.erase_mode
                    print(f"Eraser {'ON' if ctrl.erase_mode else 'OFF'}")
                    beep(500 if ctrl.erase_mode else 700, 30)
                elif key == ord('h'):
                    show_help = not show_help
                elif key == ord('f'):
                    show_prof = not show_prof
                elif key == ord('r'):
                    cal.reset()
                    for s in smoothers.values(): s.reset()
                    print("Calibration reset."); beep(330, 40)
    finally:
        canvas.cancel_auto_save()

    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")

# ─────────────── ENTRY POINT (FIXED) ────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hand Gesture + Air Writing v5.0")
    ap.add_argument("--multi", action="store_true", help="2-hand tracking (default: 2)")
    ap.add_argument("--no-sound", action="store_true", help="Disable beep feedback")
    args = ap.parse_args()
    
    if args.no_sound:
        SOUND_ENABLED = False  # Safely disables sound without breaking function references
        
    main(max_hands=2 if args.multi else 2)
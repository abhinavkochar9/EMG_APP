import os
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import mediapipe as mp
from datetime import datetime
import subprocess
import matplotlib.pyplot as plt
import random
from collections import deque
import math
import sys
import threading
import signal

# Try to avoid thread oversubscription jitter on some platforms
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# <-- Import per-exercise configuration
from config import EXERCISE_CONFIGS

# ==================================
# MODELS & CLASSES
# ==================================
class _SinePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class _TemporalConvBlock(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.dw = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=padding, groups=hidden_size)
        self.pw = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        y = x.transpose(1, 2); y = self.dw(y); y = F.gelu(y); y = self.pw(y)
        y = y.transpose(1, 2); y = self.dropout(y)
        return self.norm(y + residual)

class _SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, mult: int = 2, dropout: float = 0.1):
        super().__init__()
        inner = hidden_size * mult
        self.w1 = nn.Linear(hidden_size, inner); self.wg = nn.Linear(hidden_size, inner)
        self.w2 = nn.Linear(inner, hidden_size); self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, x):
        residual = x; v = F.silu(self.w1(x)); g = torch.sigmoid(self.wg(x))
        y = self.w2(v * g); y = self.dropout(y)
        return self.norm(y + residual)

class PoseToEMGLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2,
                 num_heads: int = 4, lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.temporal_block = _TemporalConvBlock(hidden_size, kernel_size=5, dropout=dropout)
        self.pos_enc = _SinePositionalEncoding(hidden_size)
        self.bilstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=lstm_layers,
                              batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)
        self.mha_norm = nn.LayerNorm(hidden_size)
        self.mha_dropout = nn.Dropout(dropout)
        self.ffn = _SwiGLU(hidden_size, mult=2, dropout=dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h = self.input_proj(x); h = self.temporal_block(h); h = self.pos_enc(h)
        h, _ = self.bilstm(h); attn_out, _ = self.mha(h, h, h, need_weights=False)
        h = self.mha_norm(h + self.mha_dropout(attn_out)); h = self.ffn(h)
        return self.output_layer(h)

class ForecastNet(nn.Module):
    def __init__(self, input_size=33 * 2, output_size=21 * 33 * 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, output_size)
        )
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

# ===============================================
# PRO VISUAL STYLE PACK
# ===============================================
PRO_THEME = {
    "bg": (245, 245, 245),
    "panel": (255, 255, 255),
    "panel_alt": (240, 240, 240),
    "stroke": (200, 200, 200),
    "muted": (120, 120, 120),
    "text": (30, 30, 30),
    "accent": (0, 122, 255),
    "good": (46, 204, 113),
    "warn": (241, 196, 15),
    "bad": (231, 76, 60),
    "ghost": (100, 100, 255),
}

def _aa_text(img, text, org, scale=0.6, color=PRO_THEME["text"], thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def _rounded_rect(canvas, x, y, w, h, r=14, color=PRO_THEME["panel"], thickness=-1):
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x+r, y), (x+w-r, y+h), color, thickness)
    cv2.rectangle(overlay, (x, y+r), (x+w, y+h-r), color, thickness)
    cv2.circle(overlay, (x+r, y+r), r, color, thickness)
    cv2.circle(overlay, (x+w-r, y+r), r, color, thickness)
    cv2.circle(overlay, (x+r, y+h-r), r, color, thickness)
    cv2.circle(overlay, (x+w-r, y+h-r), r, color, thickness)
    alpha=1.0 if thickness<0 else 0.9
    cv2.addWeighted(overlay, alpha, canvas, 1-alpha, 0, dst=canvas)

def _drop_shadow(canvas, x, y, w, h, r=16, blur=21, spread=6, color=(0,0,0)):
    shadow = np.zeros_like(canvas)
    _rounded_rect(shadow, x+spread, y+spread, w, h, r, color, thickness=-1)
    shadow = cv2.GaussianBlur(shadow, (blur, blur), 0)
    cv2.addWeighted(shadow, 0.35, canvas, 0.65, 0, canvas)

def draw_header_bar(width, title_left, title_right=None, sub=None):
    h = 64
    bar = np.full((h, width, 3), PRO_THEME["panel_alt"], np.uint8)
    if sub:
        _aa_text(bar, sub, (24, 22), 0.5, PRO_THEME["muted"], 1)
    _aa_text(bar, title_left, (24, 42), 0.8, PRO_THEME["text"], 2)
    if title_right:
        size = cv2.getTextSize(title_right, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        _aa_text(bar, title_right, (width - size[0] - 24, 42), 0.7, PRO_THEME["muted"], 2)
    return bar

def draw_badge(text, status="neutral"):
    color_map = {"good": PRO_THEME["good"], "warn": PRO_THEME["warn"],
                 "bad": PRO_THEME["bad"], "neutral": PRO_THEME["stroke"]}
    color = color_map.get(status, PRO_THEME["stroke"])
    padx, pady = 14, 10
    ts, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    w, h = ts[0] + padx*2, ts[1] + pady*2
    badge = np.full((h, w, 3), PRO_THEME["panel"], np.uint8)
    _rounded_rect(badge, 0, 0, w, h, r=12, color=PRO_THEME["panel"], thickness=-1)
    cv2.rectangle(badge, (0, 0), (w, h), color, 2, cv2.LINE_AA)
    _aa_text(badge, text, (padx, h - pady - 2), 0.6, color, 2)
    return badge

def draw_progress_bar(width, height, progress, label=None):
    progress = float(np.clip(progress, 0.0, 1.0))
    bar = np.full((height, width, 3), PRO_THEME["panel"], np.uint8)
    _rounded_rect(bar, 0, 0, width, height, r=height//2, color=PRO_THEME["panel"], thickness=-1)
    fill_w = int(progress * width)
    if fill_w > 0:
        overlay = bar.copy()
        _rounded_rect(overlay, 0, 0, fill_w, height, r=height//2, color=PRO_THEME["accent"], thickness=-1)
        cv2.addWeighted(overlay, 0.85, bar, 0.15, 0, bar)
    if label:
        ts, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        _aa_text(bar, label, ((width - ts[0])//2, height - 8), 0.55, PRO_THEME["bg"], 2)
    return bar

def pad_stack_h(items, pad=16, bg=PRO_THEME["bg"]):
    if not items:
        return np.full((10, 10, 3), bg, np.uint8)
    h = max(img.shape[0] for img in items)
    w = sum(img.shape[1] for img in items) + pad*(len(items)-1)
    out = np.full((h, w, 3), bg, np.uint8)
    x = 0
    for im in items:
        y = (h - im.shape[0]) // 2
        out[y:y+im.shape[0], x:x+im.shape[1]] = im
        x += im.shape[1] + pad
    return out

def pad_stack_v(items, pad=16, bg=PRO_THEME["bg"]):
    if not items:
        return np.full((10, 10, 3), bg, np.uint8)
    w = max(img.shape[1] for img in items)
    h = sum(img.shape[0] for img in items) + pad*(len(items)-1)
    out = np.full((h, w, 3), bg, np.uint8)
    y = 0
    for im in items:
        x = (w - im.shape[1]) // 2
        out[y:y+im.shape[0], x:x+im.shape[1]] = im
        y += im.shape[0] + pad
    return out

def wrap_card(content_img, title=None, width=None):
    if width and content_img.shape[1] != width:
        scale = width / content_img.shape[1]
        content_img = cv2.resize(content_img, (width, int(content_img.shape[0]*scale)))
    pad = 16
    card_w = content_img.shape[1] + pad*2
    title_h = 28 if title else 0
    card_h = content_img.shape[0] + pad*2 + title_h
    card = np.full((card_h, card_w, 3), PRO_THEME["bg"], np.uint8)
    _drop_shadow(card, 0, 0, card_w, card_h)
    _rounded_rect(card, 0, 0, card_w, card_h, r=16, color=PRO_THEME["panel"], thickness=-1)
    if title:
        _aa_text(card, title, (pad, pad + 18), 0.6, PRO_THEME["muted"], 1)
    y0 = pad + title_h
    card[y0:y0+content_img.shape[0], pad:pad+content_img.shape[1]] = content_img
    return card

def decorate_frame_bg(img):
    bg = np.full((img.shape[0]+32, img.shape[1]+32, 3), PRO_THEME["bg"], np.uint8)
    bg[16:16+img.shape[0], 16:16+img.shape[1]] = img
    return bg

def _lift_contrast(img, alpha=1.06, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# ===============================================
# GLOBAL APPLICATION CONSTANTS
# ===============================================
REPORT_DIR = "Exercise_Reports"
WINDOW_SIZE = 21
EMG_BUFFER_SIZE = 150
BUFFER_DURATION_S = 5
CALIBRATION_FRAMES = 100
TRANSITION_FRAMES = 45
REQUIRED_LANDMARKS = 30
mp_pose = mp.solutions.pose
JOINT_SPEC = {
    'Left_Shoulder':  (mp_pose.PoseLandmark.LEFT_SHOULDER.value,  'Left_Shoulder.pt'),
    'Left_Elbow':     (mp_pose.PoseLandmark.LEFT_ELBOW.value,     'Left_Elbow.pt'),
    'Left_Wrist':     (mp_pose.PoseLandmark.LEFT_WRIST.value,     'Left_Wrist.pt'),
    'Right_Shoulder': (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 'Right_Shoulder.pt'),
    'Right_Elbow':    (mp_pose.PoseLandmark.RIGHT_ELBOW.value,    'Right_Elbow.pt'),
    'Right_Wrist':    (mp_pose.PoseLandmark.RIGHT_WRIST.value,    'Right_Wrist.pt'),
}
LEFT_JOINTS  = ['Left_Shoulder', 'Left_Elbow', 'Left_Wrist']
RIGHT_JOINTS = ['Right_Shoulder','Right_Elbow','Right_Wrist']
PANEL_W, PANEL_H = 320, 180
GRAPH_COLUMN_H = PANEL_H * 2 * len(LEFT_JOINTS)
DISPLAY_WIDTH_VID = 960

BG_COLOR = PRO_THEME["bg"]
AXIS_COLOR = PRO_THEME["stroke"]
TEXT_COLOR = PRO_THEME["text"]
ACTIVATION_COLOR = PRO_THEME["accent"]
JOINT_X_COLOR = (0, 255, 100)
JOINT_Y_COLOR = (255, 100, 255)
MUSCLE_COLOR_CORRECT = PRO_THEME["good"]
MUSCLE_COLOR_INCORRECT = (160, 170, 255)
PARTICLE_COLOR_CORRECT = (240, 255, 255)
PARTICLE_COLOR_INCORRECT = (200, 210, 255)
OVERLAY_ALPHA = 0.35
BASE_THICKNESS = 58
NUM_PARTICLES_PER_LINE = 15
OFFSETS = [-7, 0, 7]
MAG_EMA_ALPHA = 0.25
CORRECT_CONSEC_FRAMES_HIDE = 15
INCORRECT_CONSEC_FRAMES_SHOW = 8

# --- New (for ghost fade timing) ---
GHOST_FADE_DURATION_S = 0.1  # <-- ghost fade in/out lasts 0.5s

# --- New (lymph node visuals) ---
LYMPH_NODE_COLOR_FILL = (200, 245, 255)     # soft cyan
LYMPH_NODE_COLOR_RING = (0, 140, 255)       # amber ring
LYMPH_NODE_RADIUS_FRAC = 0.12               # of shoulder-elbow distance
LYMPH_NODE_DOWN_OFFSET_FRAC = 0.12          # move node slightly below shoulder
LYMPH_NODE_MEDIAL_FRAC = 0.30               # shift toward chest midline

# =================================
# SESSION CONTROLLER (start / pause / stop)
# =================================
class SessionController:
    """
    Keyboard-driven session controller.
    Keys:
      S = start, Space/P = pause/resume, N = next step, B = previous step,
      E = end session early, Q = quit immediately
    """
    def __init__(self):
        self.started = False
        self.paused = False
        self.quit_now = False
        self.end_early = False
        self.skip_next = False
        self.skip_prev = False

        # Pause accounting (per step)
        self.step_pause_total = 0.0
        self._step_pause_begin = None

    def begin_step(self):
        self.step_pause_total = 0.0
        self._step_pause_begin = None

    def toggle_pause(self, audio_process=None):
        if not self.started:
            self.started = True
            self.paused = False
            return
        self.paused = not self.paused
        if self.paused:
            self._step_pause_begin = time.time()
            # stop audio process while paused
            if audio_process and audio_process.poll() is None:
                try:
                    audio_process.terminate()
                except Exception:
                    pass
        else:
            if self._step_pause_begin is not None:
                self.step_pause_total += (time.time() - self._step_pause_begin)
                self._step_pause_begin = None

    def handle_key(self, k, audio_process=None):
        # normalize to lower-case letter where applicable
        if k in (ord(' '), ord('p'), ord('P')):
            self.toggle_pause(audio_process)
        elif k in (ord('s'), ord('S')):
            self.started = True
            self.paused = False
        elif k in (ord('n'), ord('N')):
            self.skip_next = True
        elif k in (ord('b'), ord('B')):
            self.skip_prev = True
        elif k in (ord('e'), ord('E')):
            self.end_early = True
        elif k in (ord('q'), ord('Q')):
            self.quit_now = True

# =================================
# LAG FIX: NON-BLOCKING CAMERA STREAM
# =================================
class CameraStream:
    def __init__(self, src=0, width=640, height=360):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            frame = cv2.flip(frame, 1)
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return (self.frame is not None), (None if self.frame is None else self.frame.copy())

    def release(self):
        self.stopped = True
        try:
            self.t.join(timeout=0.2)
        except Exception:
            pass
        self.cap.release()

# =================================
# HELPERS
# =================================
def load_forecasting_model(path):
    model = ForecastNet()
    try:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict); model.eval()
        print(f"[INFO] Forecasting model loaded from {path}")
        return model
    except FileNotFoundError:
        print(f"[ERROR] Forecast model file not found at {path}. Exiting.")
        sys.exit()

def load_emg_models(directory):
    models = {}
    print(f"[INFO] Loading EMG models from {directory}...")
    for key, (idx, fname) in JOINT_SPEC.items():
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            try:
                m = PoseToEMGLSTM()
                state = torch.load(path, map_location='cpu')
                m.load_state_dict(state, strict=False); m.eval()
                models[key] = m
                print(f"  - Loaded model for: {key}")
            except Exception as e:
                print(f"  - [ERROR] Failed to load model for {key}: {e}")
        else:
            print(f"  - [WARN] Missing model for: {key} (at {path})")
    if not models:
        print("[ERROR] No EMG models were loaded. EMG analysis will be disabled.")
    return models

def put_label(img, text, x, y, scale=0.45, color=TEXT_COLOR, thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# Polished graphs
def create_activation_graph(mag_series, title):
    w, h = PANEL_W, PANEL_H
    img = np.full((h, w, 3), PRO_THEME["panel"], np.uint8)
    for i in range(5):
        y = int((i/4) * (h-30)) + 20
        cv2.line(img, (16, y), (w-16, y), PRO_THEME["stroke"], 1, cv2.LINE_AA)
    for i in range(6):
        x = int((i/5) * (w-32)) + 16
        cv2.line(img, (x, 20), (x, h-10), PRO_THEME["stroke"], 1, cv2.LINE_AA)
    _aa_text(img, f"{title.replace('_',' ')} • Activation", (16, 18), 0.55, PRO_THEME["muted"], 1)

    if len(mag_series) > 1:
        pts = []
        seq = list(mag_series)[-EMG_BUFFER_SIZE:]
        for i, v in enumerate(seq):
            x = 16 + int(i / max(1, EMG_BUFFER_SIZE-1) * (w-32))
            v = float(np.clip(v, 0.0, 1.0))
            y = (h-12) - int(v * (h-40))
            pts.append((x, y))
        overlay = img.copy()
        poly = np.array([(16, h-12)] + pts + [(pts[-1][0], h-12)], dtype=np.int32)
        cv2.fillPoly(overlay, [poly], ACTIVATION_COLOR)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        cv2.polylines(img, [np.array(pts)], False, (255,255,255), 2, cv2.LINE_AA)

    _aa_text(img, "1.0", (w-46, 26), 0.45, PRO_THEME["muted"], 1)
    _aa_text(img, "0.0", (w-46, h-14), 0.45, PRO_THEME["muted"], 1)
    return img

def create_joint_pos_graph(x_data, y_data, title):
    w, h = PANEL_W, PANEL_H
    img = np.full((h, w, 3), PRO_THEME["panel"], np.uint8)
    for i in range(5):
        y = int((i/4) * (h-30)) + 20
        cv2.line(img, (16, y), (w-16, y), PRO_THEME["stroke"], 1, cv2.LINE_AA)
    for i in range(6):
        x = int((i/5) * (w-32)) + 16
        cv2.line(img, (x, 20), (x, h-10), PRO_THEME["stroke"], 1, cv2.LINE_AA)
    _aa_text(img, f"{title.replace('_',' ')} • Position", (16, 18), 0.55, PRO_THEME["muted"], 1)

    if len(x_data) > 1:
        pts_x, pts_y = [], []
        xs = list(x_data)[-EMG_BUFFER_SIZE:]
        ys = list(y_data)[-EMG_BUFFER_SIZE:]
        for i in range(min(len(xs), EMG_BUFFER_SIZE)):
            x = 16 + int(i / max(1, EMG_BUFFER_SIZE-1) * (w-32))
            yx = (h-12) - int(float(np.clip(xs[i], 0, 1)) * (h-40))
            yy = (h-12) - int(float(np.clip(ys[i], 0, 1)) * (h-40))
            pts_x.append((x, yx)); pts_y.append((x, yy))
        cv2.polylines(img, [np.array(pts_x)], False, JOINT_X_COLOR, 2, cv2.LINE_AA)
        cv2.polylines(img, [np.array(pts_y)], False, JOINT_Y_COLOR, 2, cv2.LINE_AA)

    cv2.rectangle(img, (w-112, 10), (w-96, 26), JOINT_X_COLOR, -1)
    _aa_text(img, "X", (w-90, 24), 0.5, PRO_THEME["muted"], 1)
    cv2.rectangle(img, (w-64, 10), (w-48, 26), JOINT_Y_COLOR, -1)
    _aa_text(img, "Y", (w-42, 24), 0.5, PRO_THEME["muted"], 1)
    return img

def get_pose_center_and_scale(points):
    ls = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]; rs = points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lh = points[mp_pose.PoseLandmark.LEFT_HIP.value]; rh = points[mp_pose.PoseLandmark.RIGHT_HIP.value]
    mid_shoulder = (ls + rs) / 2; mid_hip = (lh + rh) / 2
    center = (mid_shoulder + mid_hip) / 2
    scale = np.linalg.norm(mid_shoulder - mid_hip)
    return center, scale

def normalize_pose_for_ghost(points):
    center, scale = get_pose_center_and_scale(points)
    return (points - center) / max(scale, 1e-5)

def normalize_point_for_emg(pt_coords, mid_shoulder, mid_hip):
    center = (mid_shoulder + mid_hip) / 2.0
    scale = np.linalg.norm(mid_shoulder - mid_hip)
    return (pt_coords - center) / max(scale, 1e-6) if scale > 1e-6 else None

def ema(prev_ema, new_value, alpha):
    return new_value if prev_ema is None else (alpha * new_value + (1 - alpha) * prev_ema)

def denormalize_single_pose(normalized_pose, live_center, live_scale, scale_ratios, frame_shape):
    ghost_scaled = normalized_pose.copy()
    ghost_scaled[:, 0] *= np.mean([scale_ratios['sw'], scale_ratios['hw']])
    ghost_scaled[:, 1] *= scale_ratios['th']
    ghost_pixel = (ghost_scaled * live_scale + live_center) * np.array([frame_shape[1], frame_shape[0]])
    return ghost_pixel

def play_audio(path, start_sec=0.0):
    """
    Play step audio using ffplay. If start_sec > 0, seek to that position.
    Returns subprocess.Popen or None if launch fails.
    """
    try:
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
        if start_sec > 0:
            cmd += ["-ss", f"{start_sec:.2f}"]
        cmd += [path]
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid if os.name != "nt" else None)
    except Exception:
        return None

def stop_audio(proc):
    if not proc:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass

class ArmParticle:
    def __init__(self, start, end, speed, jitter_amount=2):
        self.start, self.end = np.array(start, dtype=np.float32), np.array(end, dtype=np.float32)
        self.t, self.speed, self.radius, self.jitter_amount = random.uniform(0, 0.3), speed, 2, jitter_amount
    def update(self):
        self.t += self.speed
        if self.t > 1: self.t = 0
        pos = (1 - self.t) * self.start + self.t * self.end
        vec = self.end - self.start
        if np.linalg.norm(vec) != 0:
            perp = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec) * random.uniform(-self.jitter_amount, self.jitter_amount)
            pos += perp
        return pos.astype(int)

def create_particle_streams():
    return {seg: [[ArmParticle([0,0], [0,0], speed=random.uniform(0.008, 0.012)) for _ in range(NUM_PARTICLES_PER_LINE)] for _ in OFFSETS]
            for seg in ["Left_Forearm", "Left_Upper_Arm", "Right_Forearm", "Right_Upper_Arm"]}

def draw_smooth_muscle(start, end, overlay, base_thickness, color, steps=20):
    # Ensure numpy arrays (prevents "can't multiply sequence by non-int of type 'float'")
    start = np.asarray(start, dtype=np.float32)
    end   = np.asarray(end,   dtype=np.float32)

    vec = end - start
    n = np.linalg.norm(vec)
    if n == 0:
        return
    perp = np.array([-vec[1], vec[0]], dtype=np.float32) / n

    points = []
    for i in range(steps + 1):
        t = i / float(steps)
        center = (1.0 - t) * start + t * end
        thickness = float(base_thickness) * (1.0 - abs(t - 0.5) * 1.5)
        points.append((center + perp * thickness).astype(np.int32))
    for i in reversed(range(steps + 1)):
        t = i / float(steps)
        center = (1.0 - t) * start + t * end
        thickness = float(base_thickness) * (1.0 - abs(t - 0.5) * 1.5)
        points.append((center - perp * thickness).astype(np.int32))

    cv2.fillPoly(overlay, [np.array(points)], color)

def get_parallel_start_end(start, end, offset):
    start = np.asarray(start, dtype=np.float32)
    end   = np.asarray(end,   dtype=np.float32)
    vec = end - start
    n = np.linalg.norm(vec)
    if n == 0:
        return start, end
    perp = np.array([-vec[1], vec[0]], dtype=np.float32) / n * float(offset)
    return start + perp, end + perp

def calculate_similarity_scores(user_landmarks, target_landmarks, frame_shape):
    ARM_SEGMENTS = {
        "Left_Upper_Arm": (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value),
        "Left_Forearm": (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
        "Right_Upper_Arm": (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
        "Right_Forearm": (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value)
    }
    similarity_scores = []; frame_h, frame_w, _ = frame_shape
    for segment_name, (p1_idx, p2_idx) in ARM_SEGMENTS.items():
        user_p1, user_p2 = user_landmarks[p1_idx], user_landmarks[p2_idx]
        target_p1, target_p2 = target_landmarks[p1_idx], target_landmarks[p2_idx]
        dist1 = np.linalg.norm((user_p1 - target_p1) / np.array([frame_w, frame_h]))
        dist2 = np.linalg.norm((user_p2 - target_p2) / np.array([frame_w, frame_h]))
        similarity = max(0, 100 * (1 - (dist1 + dist2) / 2))
        similarity_scores.append({'segment': segment_name, 'score': similarity, 'p1_idx': p1_idx, 'p2_idx': p2_idx})
    return similarity_scores

# --- NEW: lymph node computation helper ---
def compute_lymph_node(landmarks_pixels):
    """
    Returns dict:
      {
        'Left':  {'center': np.array([x,y]), 'radius': int},
        'Right': {'center': np.array([x,y]), 'radius': int}
      }
    """
    ls = landmarks_pixels[mp_pose.PoseLandmark.LEFT_SHOULDER.value].astype(float)
    rs = landmarks_pixels[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].astype(float)
    le = landmarks_pixels[mp_pose.PoseLandmark.LEFT_ELBOW.value].astype(float)
    re = landmarks_pixels[mp_pose.PoseLandmark.RIGHT_ELBOW.value].astype(float)

    mid_x = (ls[0] + rs[0]) / 2.0

    # Left side
    dist_left = np.linalg.norm(le - ls) + 1e-6
    left_radius = int(np.clip(dist_left * LYMPH_NODE_RADIUS_FRAC, 10, 34))
    left_center = ls.copy()
    left_center[1] += dist_left * LYMPH_NODE_DOWN_OFFSET_FRAC
    left_center[0] += (mid_x - left_center[0]) * LYMPH_NODE_MEDIAL_FRAC  # shift toward chest

    # Right side
    dist_right = np.linalg.norm(re - rs) + 1e-6
    right_radius = int(np.clip(dist_right * LYMPH_NODE_RADIUS_FRAC, 10, 34))
    right_center = rs.copy()
    right_center[1] += dist_right * LYMPH_NODE_DOWN_OFFSET_FRAC
    right_center[0] += (mid_x - right_center[0]) * LYMPH_NODE_MEDIAL_FRAC  # shift toward chest

    return {
        'Left':  {'center': left_center,  'radius': left_radius},
        'Right': {'center': right_center, 'radius': right_radius}
    }

# <-- UPDATED: particles + ribbons target the lymph node at shoulders
def draw_muscle_flow_feedback(frame, user_landmarks, scores, base_thickness, particle_streams, threshold):
    overlay = frame.copy()

    # Compute lymph node positions (slightly below & medial to shoulders)
    nodes = compute_lymph_node(user_landmarks)

    # Draw filled lymph nodes on overlay (so they appear under particles)
    for side in ('Left', 'Right'):
        c = nodes[side]['center'].astype(np.int32)
        r = int(nodes[side]['radius'])
        cv2.circle(overlay, tuple(c), r, LYMPH_NODE_COLOR_FILL, -1, cv2.LINE_AA)

    for seg_data in scores:
        side = 'Left' if seg_data['segment'].startswith('Left') else 'Right'
        node_center = np.asarray(nodes[side]['center'], dtype=np.float32)

        is_correct = seg_data['score'] > threshold
        muscle_color = MUSCLE_COLOR_CORRECT if is_correct else MUSCLE_COLOR_INCORRECT
        particle_color = PARTICLE_COLOR_CORRECT if is_correct else PARTICLE_COLOR_INCORRECT

        # Indices per side
        elbow_idx = (mp_pose.PoseLandmark.LEFT_ELBOW.value
                     if side == 'Left' else mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        wrist_idx = (mp_pose.PoseLandmark.LEFT_WRIST.value
                     if side == 'Left' else mp_pose.PoseLandmark.RIGHT_WRIST.value)

        elbow_pt = np.asarray(user_landmarks[elbow_idx], dtype=np.float32)
        wrist_pt = np.asarray(user_landmarks[wrist_idx], dtype=np.float32)

        if 'Upper_Arm' in seg_data['segment']:
            # Upper arm: ribbon & particles from ELBOW → LYMPH NODE
            p_start = elbow_pt
            p_end   = node_center
            draw_smooth_muscle(p_end, p_start, overlay, base_thickness, muscle_color)  # draw toward node

            dynamic_offsets = [float(offset) * (float(base_thickness) / float(BASE_THICKNESS)) for offset in OFFSETS]
            for i, offset in enumerate(dynamic_offsets):
                s, e = get_parallel_start_end(p_start, p_end, offset)
                # Land particles inside node
                r = float(nodes[side]['radius'])
                jitter = (np.random.uniform(-0.4, 0.4, size=2).astype(np.float32)) * r
                e = node_center + jitter
                for particle in particle_streams[seg_data['segment']][i]:
                    particle.start = np.asarray(s, dtype=np.float32)
                    particle.end   = np.asarray(e, dtype=np.float32)
                    pos = particle.update() if is_correct else (1 - particle.t) * particle.start + particle.t * particle.end
                    cv2.circle(frame, tuple(pos.astype(int)), particle.radius, particle_color, -1, cv2.LINE_AA)

        else:
            # Forearm: ribbon & particles from WRIST → ELBOW (forearm should connect to elbow, not shoulder)
            p_start = wrist_pt
            p_end   = elbow_pt
            draw_smooth_muscle(p_start, p_end, overlay, base_thickness * 0.85, muscle_color)  # slightly thinner

            dynamic_offsets = [float(offset) * (float(base_thickness) / float(BASE_THICKNESS)) for offset in OFFSETS]
            for i, offset in enumerate(dynamic_offsets):
                s, e = get_parallel_start_end(p_start, p_end, offset)
                # Land particles just inside an elbow "hub"
                forearm_len = max(1.0, float(np.linalg.norm(p_end - p_start)))
                elbow_hub_r = max(6.0, min(16.0, forearm_len * 0.06))
                jitter = (np.random.uniform(-0.35, 0.35, size=2).astype(np.float32)) * elbow_hub_r
                e = elbow_pt + jitter
                for particle in particle_streams[seg_data['segment']][i]:
                    particle.start = np.asarray(s, dtype=np.float32)
                    particle.end   = np.asarray(e, dtype=np.float32)
                    pos = particle.update() if is_correct else (1 - particle.t) * particle.start + particle.t * particle.end
                    cv2.circle(frame, tuple(pos.astype(int)), particle.radius, particle_color, -1, cv2.LINE_AA)

    # Blend overlay (ribbons + filled nodes) with base frame (which already has particles)
    blended = cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0)

    # Draw crisp lymph node rings on top to highlight the nodes
    for side in ('Left', 'Right'):
        c = nodes[side]['center'].astype(np.int32)
        r = int(nodes[side]['radius'])
        cv2.circle(blended, tuple(c), r, LYMPH_NODE_COLOR_RING, 2, cv2.LINE_AA)
        cv2.circle(blended, tuple(c), max(2, r // 3), LYMPH_NODE_COLOR_RING, 1, cv2.LINE_AA)

    # Optional: draw a small elbow hub so the forearm visually anchors there
    for side in ('Left', 'Right'):
        elbow_idx = (mp_pose.PoseLandmark.LEFT_ELBOW.value
                     if side == 'Left' else mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        elbow_pt = np.asarray(user_landmarks[elbow_idx], dtype=np.float32).astype(np.int32)
        cv2.circle(blended, tuple(elbow_pt), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(blended, tuple(elbow_pt), 7, PRO_THEME["stroke"], 1, cv2.LINE_AA)

    return blended

def draw_ghost(frame, ghost_landmarks, color=PRO_THEME["ghost"], opacity=0.35):
    ARM_SEGMENTS = [(11, 13), (13, 15), (12, 14), (14, 16)]
    overlay = frame.copy()
    for p1_idx, p2_idx in ARM_SEGMENTS:
        if p1_idx < len(ghost_landmarks) and p2_idx < len(ghost_landmarks):
            p1, p2 = ghost_landmarks[p1_idx], ghost_landmarks[p2_idx]
            if not (np.isnan(p1).any() or np.isnan(p2).any()):
                cv2.line(overlay, tuple(np.int32(p1)), tuple(np.int32(p2)), color, 8, cv2.LINE_AA)
    return cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

def check_neck_posture(landmarks, frame):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
    neck_offset_x = nose[0] - shoulder_mid_x
    threshold = (left_shoulder[0] - right_shoulder[0]) * 0.15
    if abs(neck_offset_x) > threshold:
        cv2.putText(frame, "ADJUST NECK", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Neck Posture: GOOD", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def pose_is_correct(scores, threshold):
    return all(s['score'] >= threshold for s in scores)

def draw_controls_hint(w):
    h = 92
    panel = np.full((h, w, 3), PRO_THEME["panel"], np.uint8)
    txt = "Controls: S=start  Space/P=pause/resume  N=next  B=prev  E=end  Q=quit"
    _aa_text(panel, txt, (18, 58), 0.7, PRO_THEME["muted"], 2)
    return wrap_card(panel)

# ==================================
# EXERCISE SESSION RUNNER (with start/pause/stop)
# ==================================
def run_exercise_session(config):
    start_time = datetime.now()
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Session controller
    ctrl = SessionController()

    # Faster MediaPipe configuration
    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True
    )

    print(f"\n[INFO] Initializing exercise: {config['name']}")
    emg_models = load_emg_models(config['emg_models_dir'])
    forecast_model = load_forecasting_model(config['forecast_model_path'])
    particle_streams = create_particle_streams()

    # Load step videos
    try:
        step_videos = [cv2.VideoCapture(os.path.join(config['step_vid_dir'], f"Step{i + 1}.mp4"))
                       for i in range(len(config['step_keyframe_indices']))]
        if not all(v.isOpened() for v in step_videos):
            raise IOError("One or more step videos failed to open.")
    except Exception as e:
        print(f"[ERROR] Could not load step videos from {config['step_vid_dir']}: {e}")
        return

    # Non-blocking webcam (smaller capture; upscale later)
    cam = CameraStream(src=0, width=640, height=360)

    # Calibration
    print("[INFO] Starting calibration...")
    calibration_poses = []
    while len(calibration_poses) < CALIBRATION_FRAMES:
        ret, frame = cam.read()
        if not ret:
            continue
        results = pose_estimator.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks and results.pose_landmarks.landmark:
            landmarks = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark])
            if np.sum((landmarks[:, 0] >= 0) & (landmarks[:, 0] <= 1)) >= REQUIRED_LANDMARKS:
                calibration_poses.append(landmarks)
        progress_text = f"Calibrating... ({len(calibration_poses)}/{CALIBRATION_FRAMES})"
        cv2.putText(frame, progress_text, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("Calibration", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q')):
            cam.release()
            cv2.destroyAllWindows()
            return
    cv2.destroyWindow("Calibration")

    if not calibration_poses:
        print("[ERROR] No calibration poses captured.")
        cam.release()
        return

    avg_calib_pose_norm = np.mean([normalize_pose_for_ghost(p) for p in calibration_poses[-20:]], axis=0)
    with torch.inference_mode():
        predicted_kf_normalized = forecast_model(torch.tensor(avg_calib_pose_norm, dtype=torch.float32).unsqueeze(0)).numpy().reshape(-1, 33, 2)

    # Scale ratios between reference ghost and user
    ref_pose_norm = predicted_kf_normalized[0]
    sw_ref = np.linalg.norm(ref_pose_norm[11] - ref_pose_norm[12])
    hw_ref = np.linalg.norm(ref_pose_norm[23] - ref_pose_norm[24])
    th_ref = np.linalg.norm((ref_pose_norm[11:13].mean(axis=0)) - (ref_pose_norm[23:25].mean(axis=0)))
    sw_user = np.linalg.norm(avg_calib_pose_norm[11] - avg_calib_pose_norm[12])
    hw_user = np.linalg.norm(avg_calib_pose_norm[23] - avg_calib_pose_norm[24])
    th_user = np.linalg.norm((avg_calib_pose_norm[11:13].mean(axis=0)) - (avg_calib_pose_norm[23:25].mean(axis=0)))
    scale_ratios = {
        'sw': sw_user / max(sw_ref, 1e-5),
        'hw': hw_user / max(hw_ref, 1e-5),
        'th': th_user / max(th_ref, 1e-5)
    }

    # Buffers
    performance_data, step_counter, repetitions = [], 0, 0
    prev_norm_pose, last_valid_ghost = avg_calib_pose_norm, None
    ghost_visible, correct_run, incorrect_run = True, 0, 0
    fading_out, fading_in, fade_start_time = False, False, None  # <-- time-based fade
    joint_input_buffers = { key: deque(maxlen=WINDOW_SIZE) for key in JOINT_SPEC.keys() }
    trace_buffers = { key: {'x': deque(maxlen=EMG_BUFFER_SIZE), 'y': deque(maxlen=EMG_BUFFER_SIZE)} for key in JOINT_SPEC.keys() }
    emg_buffers = { key: { 'ch1': deque(maxlen=EMG_BUFFER_SIZE), 'ch2': deque(maxlen=EMG_BUFFER_SIZE), 'mag': deque(maxlen=EMG_BUFFER_SIZE), 'mag_ema_last': None } for key in JOINT_SPEC.keys() }

    # Throttling for lag reduction
    frame_idx = 0
    EMG_EVERY = 2
    GRAPHS_EVERY = 3
    left_graphs_img = None
    right_graphs_img = None

    # Wait splash to press S to start
    splash = np.full((280, 1000, 3), PRO_THEME["panel"], np.uint8)
    _aa_text(splash, "Ready to start session", (40, 110), 1.2, PRO_THEME["text"], 2)
    _aa_text(splash, "Press S to start • Q to quit", (40, 170), 0.9, PRO_THEME["muted"], 2)
    cv2.imshow("Integrated 3D Exercise and EMG-Based Muscle Activity Visualization", splash)
    while not ctrl.started:
        k = cv2.waitKey(30) & 0xFF
        ctrl.handle_key(k)
        if ctrl.quit_now:
            cam.release()
            cv2.destroyAllWindows()
            return

    print("[INFO] Starting exercise...")
    while repetitions < config['total_reps']:
        # setup per-step timing and state
        target_kf_idx = config['step_keyframe_indices'][step_counter]
        target_norm_pose = predicted_kf_normalized[target_kf_idx]
        print(f"[INFO] Rep {repetitions + 1}/{config['total_reps']}: Starting Step {step_counter + 1}...")

        step_video = step_videos[step_counter]
        step_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        video_fps = step_video.get(cv2.CAP_PROP_FPS)
        if not video_fps or video_fps <= 0:
            video_fps = 30

        audio_path = os.path.join(config['step_vid_dir'], f"Step{step_counter + 1}.mp4")
        ctrl.begin_step()
        step_start_time = time.time()
        audio_process = play_audio(audio_path, start_sec=0.0)

        while True:
            # keyboard handling (always responsive)
            k = cv2.waitKey(1) & 0xFF
            if k != 255:
                ctrl.handle_key(k, audio_process)

            if ctrl.quit_now:
                stop_audio(audio_process)
                cam.release()
                for v in step_videos: v.release()
                cv2.destroyAllWindows()
                return
            if ctrl.end_early:
                stop_audio(audio_process)
                break
            if ctrl.skip_next:
                ctrl.skip_next = False
                stop_audio(audio_process)
                prev_norm_pose = target_norm_pose
                step_counter = (step_counter + 1) % len(config['step_keyframe_indices'])
                break
            if ctrl.skip_prev:
                ctrl.skip_prev = False
                stop_audio(audio_process)
                prev_norm_pose = target_norm_pose
                step_counter = (step_counter - 1) % len(config['step_keyframe_indices'])
                break

            # compute elapsed time excluding pauses
            raw_elapsed = time.time() - step_start_time
            effective_elapsed = raw_elapsed - ctrl.step_pause_total

            # Handle pause
            if ctrl.paused:
                paused_panel = np.full((180, 900, 3), PRO_THEME["panel"], np.uint8)
                _aa_text(paused_panel, "Paused", (40, 90), 1.2, PRO_THEME["text"], 2)
                _aa_text(paused_panel, "Press Space/P to resume • E end • Q quit", (40, 140), 0.8, PRO_THEME["muted"], 2)
                cv2.imshow("Integrated 3D Exercise and EMG-Based Muscle Activity Visualization", decorate_frame_bg(wrap_card(paused_panel)))
                continue
            else:
                if audio_process is None or (audio_process and audio_process.poll() is not None):
                    audio_process = play_audio(audio_path, start_sec=effective_elapsed)

            # Time-synced reference video frame
            target_frame_index = int(effective_elapsed * video_fps)
            step_video.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
            ret_vid, frame_vid = step_video.read()
            if not ret_vid:
                break

            # Latest webcam frame (non-blocking)
            ret_user, frame_user = cam.read()
            if not ret_user:
                continue

            results = pose_estimator.process(cv2.cvtColor(frame_user, cv2.COLOR_BGR2RGB))

            # Layout sizing (we upscale webcam to match visual column height)
            h, w, _ = frame_user.shape
            new_user_width = int(GRAPH_COLUMN_H * (w / h))
            display_frame = cv2.resize(frame_user, (new_user_width, GRAPH_COLUMN_H))

            scores = []
            frame_idx += 1

            if results.pose_landmarks and results.pose_landmarks.landmark:
                lm = results.pose_landmarks.landmark
                lsh,rsh = np.array([lm[11].x,lm[11].y]), np.array([lm[12].x,lm[12].y])
                lhip,rhip = np.array([lm[23].x,lm[23].y]), np.array([lm[24].x,lm[24].y])
                mid_shoulder, mid_hip = (lsh + rsh) / 2.0, (lhip + rhip) / 2.0

                # Update traces every frame (cheap)
                for key, (idx, _) in JOINT_SPEC.items():
                    j = lm[idx]
                    if j.visibility > 0.5:
                        trace_buffers[key]['x'].append(j.x); trace_buffers[key]['y'].append(j.y)
                        norm_pt = normalize_point_for_emg(np.array([j.x, j.y]), mid_shoulder, mid_hip)
                        if norm_pt is not None: joint_input_buffers[key].append(norm_pt)

                # EMG inference less frequently
                if (frame_idx % EMG_EVERY) == 0:
                    for key, model in emg_models.items():
                        if len(joint_input_buffers[key]) == WINDOW_SIZE:
                            x_tensor = torch.tensor([list(joint_input_buffers[key])], dtype=torch.float32)
                            with torch.inference_mode():
                                y_last = model(x_tensor)[:, -1, :]
                            ch1, ch2 = y_last.squeeze(0).cpu().numpy().tolist()
                            emg_buffers[key]['ch1'].append(ch1); emg_buffers[key]['ch2'].append(ch2)
                            mag_n = math.sqrt(ch1**2 + ch2**2) / (1.0 + math.sqrt(ch1**2 + ch2**2))
                            smoothed = ema(emg_buffers[key]['mag_ema_last'], mag_n, MAG_EMA_ALPHA)
                            emg_buffers[key]['mag_ema_last'] = smoothed
                            emg_buffers[key]['mag'].append(smoothed)

                # Ghost mapping + similarity
                user_landmarks_norm = np.array([[l.x, l.y] for l in lm])
                live_center, live_scale = get_pose_center_and_scale(user_landmarks_norm)
                user_landmarks_pixels = user_landmarks_norm * np.array([display_frame.shape[1], display_frame.shape[0]])

                # transition progress uses effective_elapsed (excludes pauses)
                alpha = min(1.0, effective_elapsed / (TRANSITION_FRAMES / max(video_fps, 1)))
                current_norm_ghost = (1 - alpha) * prev_norm_pose + alpha * target_norm_pose

                # Enlarge the ghost figure slightly
                ghost_scale_factor = 1.05  # Increase ghost size by 5%
                enlarged_live_scale = live_scale * ghost_scale_factor
                dynamic_ghost_pose = denormalize_single_pose(current_norm_ghost, live_center, enlarged_live_scale, scale_ratios, display_frame.shape[:2])
                
                last_valid_ghost = dynamic_ghost_pose
                scores = calculate_similarity_scores(user_landmarks_pixels, dynamic_ghost_pose, display_frame.shape)

                if pose_is_correct(scores, config['similarity_threshold']):
                    correct_run += 1; incorrect_run = 0
                else:
                    incorrect_run += 1; correct_run = 0

                # Ghost visibility logic (time-based 0.5s fade)
                now = time.time()
                if ghost_visible and not fading_out and correct_run >= CORRECT_CONSEC_FRAMES_HIDE:
                    fading_out, fading_in, fade_start_time = True, False, now
                if not ghost_visible and not fading_in and incorrect_run >= INCORRECT_CONSEC_FRAMES_SHOW:
                    fading_in, fading_out, fade_start_time = True, False, now

                if fading_out:
                    elapsed = now - fade_start_time
                    opacity = max(0.0, OVERLAY_ALPHA * (1.0 - (elapsed / GHOST_FADE_DURATION_S)))
                    display_frame = draw_ghost(display_frame, dynamic_ghost_pose, opacity=opacity)
                    if elapsed >= GHOST_FADE_DURATION_S:
                        fading_out, ghost_visible = False, False
                elif fading_in:
                    elapsed = now - fade_start_time
                    opacity = min(OVERLAY_ALPHA, OVERLAY_ALPHA * (elapsed / GHOST_FADE_DURATION_S))
                    display_frame = draw_ghost(display_frame, dynamic_ghost_pose, opacity=opacity)
                    if elapsed >= GHOST_FADE_DURATION_S:
                        fading_in, ghost_visible = False, True
                elif ghost_visible:
                    display_frame = draw_ghost(display_frame, dynamic_ghost_pose, opacity=OVERLAY_ALPHA)

                # Muscle ribbons + particles flowing into lymph node (shoulder)
                scale_ratio = live_scale / max(1e-5, np.linalg.norm(avg_calib_pose_norm[11:13].mean(axis=0) - avg_calib_pose_norm[23:25].mean(axis=0)))
                dynamic_thickness = int(np.clip(BASE_THICKNESS * scale_ratio, 8, 44))
                display_frame = draw_muscle_flow_feedback(display_frame, user_landmarks_pixels, scores, dynamic_thickness, particle_streams, config['similarity_threshold'])

                # Neck posture text
                check_neck_posture(user_landmarks_pixels, display_frame)

                # Save performance rows
                for score_data in scores:
                    performance_data.append({
                        'timestamp': datetime.now(),
                        'repetition': repetitions + 1,
                        'step': step_counter + 1,
                        'segment': score_data['segment'],
                        'score': score_data['score']
                    })

            elif last_valid_ghost is not None and (ghost_visible or fading_in or fading_out):
                # still show fading ghost if we lost landmarks
                now = time.time()
                if fading_out:
                    elapsed = now - fade_start_time
                    opacity = max(0.0, OVERLAY_ALPHA * (1.0 - (elapsed / GHOST_FADE_DURATION_S)))
                elif fading_in:
                    elapsed = now - fade_start_time
                    opacity = min(OVERLAY_ALPHA, OVERLAY_ALPHA * (elapsed / GHOST_FADE_DURATION_S))
                else:
                    opacity = OVERLAY_ALPHA
                display_frame = draw_ghost(display_frame, last_valid_ghost, opacity=opacity)

            # Rebuild graphs only sometimes; otherwise reuse (lag fix)
            def build_joint_column_once(joint_keys):
                rows = []
                for k2 in joint_keys:
                    a = create_activation_graph(list(emg_buffers[k2]['mag']), k2)
                    p = create_joint_pos_graph(list(trace_buffers[k2]['x']), list(trace_buffers[k2]['y']), k2)
                    rows.append(np.vstack((a, p)))
                return np.vstack(rows)

            if (frame_idx % GRAPHS_EVERY) == 0 or left_graphs_img is None:
                left_graphs_img  = build_joint_column_once(LEFT_JOINTS)
                right_graphs_img = build_joint_column_once(RIGHT_JOINTS)

            left_graphs = left_graphs_img
            right_graphs = right_graphs_img

            # Visual polish: contrast lift (cheap)
            frame_vid = _lift_contrast(frame_vid, 1.06, 0)
            display_frame = _lift_contrast(display_frame, 1.06, 0)

            # 4-COLUMN LAYOUT: Reference | Live | Left Graphs | Right Graphs
            frame_vid_resized = cv2.resize(frame_vid, (DISPLAY_WIDTH_VID, GRAPH_COLUMN_H))
            ref_col   = wrap_card(frame_vid_resized, title="Reference")
            live_col  = wrap_card(display_frame,      title="You")
            left_col  = wrap_card(left_graphs,        title="Left Arm")
            right_col = wrap_card(right_graphs,       title="Right Arm")

            four_col_row = pad_stack_h([ref_col, live_col, left_col, right_col], pad=16, bg=PRO_THEME["bg"])

            # Badges + timeline + controls hint
            if results and results.pose_landmarks and results.pose_landmarks.landmark and scores:
                mean_score = float(np.mean([s['score'] for s in scores]))
                status = "good" if mean_score >= (config['similarity_threshold'] + 5) else ("warn" if mean_score >= config['similarity_threshold'] else "bad")
                badge = draw_badge(f"Similarity {mean_score:0.0f}%", status=status)
            else:
                badge = draw_badge("Searching pose…", status="neutral")
            rep_progress = (step_counter) / max(1, len(config['step_keyframe_indices']))
            timeline = draw_progress_bar(420, 24, rep_progress, label="step timeline")
            timeline_card = wrap_card(timeline, title=None)
            controls_card = draw_controls_hint(640)
            badges_row = pad_stack_h([badge, timeline_card, controls_card], pad=16, bg=PRO_THEME["bg"])

            # Header
            body = pad_stack_v([four_col_row, badges_row], pad=16, bg=PRO_THEME["bg"])
            header_w = body.shape[1]
            left_title = f"{config['name']} • Step {step_counter + 1}/{len(config['step_keyframe_indices'])}"
            right_title = f"Rep {repetitions + 1}/{config['total_reps']}"
            sub = "Posture Coach — Live Session (S start • Space/P pause • N next • B prev • E end • Q quit)"
            header = draw_header_bar(header_w, title_left=left_title, title_right=right_title, sub=sub)

            canvas = pad_stack_v([header, body], pad=12, bg=PRO_THEME["bg"])
            canvas = decorate_frame_bg(canvas)

            cv2.imshow("Integrated 3D Exercise and EMG-Based Muscle Activity Visualization", canvas)

        # end while (step)
        stop_audio(audio_process)
        if ctrl.quit_now:
            break
        if ctrl.end_early:
            print("[INFO] Ending session early by user request.")
            break

        prev_norm_pose = target_norm_pose
        step_counter += 1
        if step_counter >= len(config['step_keyframe_indices']):
            step_counter, repetitions = 0, repetitions + 1

    print(f"[INFO] Exercise '{config['name']}' complete!")
    end_time = datetime.now()
    if performance_data:
        print("[INFO] Generating performance report and graph...")
        df = pd.DataFrame(performance_data)
        total_duration, avg_score = (end_time - start_time).total_seconds(), df['score'].mean()
        avg_by_rep, avg_by_step = df.groupby('repetition')['score'].mean(), df.groupby(['repetition', 'step'])['score'].mean()

        os.makedirs(REPORT_DIR, exist_ok=True)
        report_filename = os.path.join(REPORT_DIR, f"Report_{config['name']}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
        graph_filename = os.path.join(REPORT_DIR, f"Graph_{config['name']}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}.png")

        with open(report_filename, 'w') as f:
            f.write(f"Performance Report for {config['name']}\nSession Time,{start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration (s),{total_duration:.2f}\nReps Completed,{df['repetition'].max() if not df.empty else 0}/{config['total_reps']}\n")
            f.write(f"Overall Average Score,{avg_score:.2f}\n\n,Average Score by Repetition\n")
            avg_by_rep.to_csv(f, header=True)
            f.write("\n,Average Score by Step\n")
            avg_by_step.to_csv(f, header=True)
        print(f"[INFO] Report CSV saved to {report_filename}")

        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(12, 7))
            x_labels = [f"R{r}-S{s}" for r, s in avg_by_step.index]
            ax.plot(x_labels, avg_by_step.values, marker='o', linestyle='-', color='dodgerblue', label='Average Score')
            ax.axhline(y=avg_score, color='red', linestyle='--', label=f'Overall Avg: {avg_score:.2f}')
            ax.set_title(f"{config['name']} Performance Trend", fontsize=16)
            ax.set_xlabel('Exercise Progression (Rep-Step)', fontsize=12)
            ax.set_ylabel('Average Similarity Score (%)', fontsize=12)
            ax.set_ylim(0, 105); plt.xticks(rotation=45, ha='right'); ax.legend(); plt.tight_layout()
            plt.savefig(graph_filename); plt.close(fig)
            print(f"[INFO] Performance graph saved to {graph_filename}")
        except Exception as e:
            print(f"[ERROR] Could not generate graph. Matplotlib installed? Error: {e}")

    # Cleanup
    cam.release()
    for v in step_videos: v.release()
    cv2.destroyAllWindows()
    print("[INFO] Session finished.")

# ============================
# MAIN CONTROLLER
# ============================
def main():
    if len(sys.argv) > 1 and sys.argv[1] in EXERCISE_CONFIGS:
        selected_exercise = sys.argv[1]
    else:
        print("\n[ERROR] Please specify a valid exercise to run.")
        print("Usage: python main.py <ExerciseName>")
        print("Available exercises:")
        for name in EXERCISE_CONFIGS.keys():
            print(f"  - {name}")
        return

    selected_config = EXERCISE_CONFIGS[selected_exercise]
    run_exercise_session(selected_config)

if __name__ == '__main__':
    main()
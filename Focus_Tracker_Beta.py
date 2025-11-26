import cv2
import mediapipe as mp
import time
import numpy as np
from datetime import timedelta, datetime
from collections import deque
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import os
import platform
import tkinter as tk
from tkinter import messagebox

# ============================
# Ultra AI Focus Tracker v4.0
# Apple-Inspired Modern UI Design
# ============================

# Color Palette (Apple-inspired)
COLORS = {
    'bg_dark': (18, 18, 20),  # Almost black
    'bg_panel': (28, 28, 32),  # Dark gray panel
    'bg_card': (44, 44, 48),  # Card background
    'accent_blue': (0, 122, 255),  # iOS Blue
    'accent_green': (52, 199, 89),  # iOS Green
    'accent_orange': (255, 149, 0),  # iOS Orange
    'accent_red': (255, 59, 48),  # iOS Red
    'accent_purple': (175, 82, 222),  # iOS Purple
    'accent_cyan': (90, 200, 250),  # iOS Cyan
    'text_primary': (255, 255, 255),  # White
    'text_secondary': (142, 142, 147),  # Gray
    'text_tertiary': (99, 99, 102),  # Darker gray
    'success': (52, 199, 89),  # Green
    'warning': (255, 204, 0),  # Yellow
    'error': (255, 59, 48),  # Red
}


# BGR versions for OpenCV
def rgb_to_bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


COLORS_BGR = {k: rgb_to_bgr(v) for k, v in COLORS.items()}


def get_downloads_folder():
    """Get the Downloads folder path for any OS"""
    if platform.system() == "Windows":
        import winreg
        try:
            sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
                downloads = winreg.QueryValueEx(key, '{374DE290-123F-4565-9164-39C4925E467B}')[0]
                return downloads
        except:
            pass
        return os.path.join(os.path.expanduser('~'), 'Downloads')
    elif platform.system() == "Darwin":
        return os.path.join(os.path.expanduser('~'), 'Downloads')
    else:
        xdg_config = os.path.join(os.path.expanduser('~'), '.config', 'user-dirs.dirs')
        if os.path.exists(xdg_config):
            try:
                with open(xdg_config, 'r') as f:
                    for line in f:
                        if line.startswith('XDG_DOWNLOAD_DIR'):
                            path = line.split('=')[1].strip().strip('"')
                            path = path.replace('$HOME', os.path.expanduser('~'))
                            return path
            except:
                pass
        return os.path.join(os.path.expanduser('~'), 'Downloads')


def get_report_filename():
    """Generate unique filename with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return f"FocusTracker_Report_{timestamp}"


class ModernStartDialog:
    """Apple-style modern start dialog"""

    def __init__(self):
        self.should_start = False
        self.root = tk.Tk()
        self.setup_window()
        self.create_widgets()

    def setup_window(self):
        self.root.title("Focus Tracker")
        self.root.geometry("480x380")
        self.root.resizable(False, False)

        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (240)
        y = (self.root.winfo_screenheight() // 2) - (190)
        self.root.geometry(f"480x380+{x}+{y}")

        # Modern dark theme
        self.bg_color = "#111113"
        self.card_color = "#1c1c1e"
        self.fg_color = "#ffffff"
        self.accent_color = "#007AFF"
        self.secondary_color = "#8e8e93"

        self.root.configure(bg=self.bg_color)
        self.root.protocol("WM_DELETE_WINDOW", self.on_cancel)

    def create_widgets(self):
        # Main container with padding
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(expand=True, fill='both', padx=40, pady=30)

        # App icon placeholder (circle with gradient effect simulated)
        icon_frame = tk.Frame(main_frame, bg=self.bg_color)
        icon_frame.pack(pady=(0, 15))

        icon_canvas = tk.Canvas(icon_frame, width=70, height=70,
                                bg=self.bg_color, highlightthickness=0)
        icon_canvas.pack()
        icon_canvas.create_oval(5, 5, 65, 65, fill="#007AFF", outline="")
        icon_canvas.create_text(35, 35, text="🎯", font=("Segoe UI", 28))

        # Title
        title_label = tk.Label(
            main_frame,
            text="Focus Tracker",
            font=("SF Pro Display", 26, "bold") if platform.system() == "Darwin"
            else ("Segoe UI", 24, "bold"),
            fg=self.fg_color,
            bg=self.bg_color
        )
        title_label.pack(pady=(0, 5))

        # Subtitle
        subtitle_label = tk.Label(
            main_frame,
            text="AI-Powered Focus Analysis",
            font=("SF Pro Text", 13) if platform.system() == "Darwin"
            else ("Segoe UI", 11),
            fg=self.secondary_color,
            bg=self.bg_color
        )
        subtitle_label.pack(pady=(0, 25))

        # Info card
        card_frame = tk.Frame(main_frame, bg=self.card_color,
                              highlightbackground="#2c2c2e", highlightthickness=1)
        card_frame.pack(fill='x', pady=(0, 20), ipady=15, ipadx=15)

        info_icon = tk.Label(card_frame, text="📁", font=("Segoe UI", 16),
                             bg=self.card_color, fg=self.fg_color)
        info_icon.pack(anchor='w', padx=15, pady=(10, 5))

        info_text = tk.Label(
            card_frame,
            text="Your session report will be automatically\nsaved to Downloads as PDF & PNG",
            font=("SF Pro Text", 11) if platform.system() == "Darwin"
            else ("Segoe UI", 10),
            fg=self.secondary_color,
            bg=self.card_color,
            justify='left'
        )
        info_text.pack(anchor='w', padx=15, pady=(0, 10))

        # Buttons frame
        button_frame = tk.Frame(main_frame, bg=self.bg_color)
        button_frame.pack(fill='x', pady=(10, 0))

        # Cancel button (secondary style)
        cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            font=("SF Pro Text", 13) if platform.system() == "Darwin"
            else ("Segoe UI", 11),
            bg="#2c2c2e",
            fg=self.fg_color,
            activebackground="#3a3a3c",
            activeforeground=self.fg_color,
            relief='flat',
            cursor='hand2',
            width=12,
            height=2,
            command=self.on_cancel
        )
        cancel_btn.pack(side='left', padx=(0, 10))

        # Start button (primary style)
        start_btn = tk.Button(
            button_frame,
            text="Start Session",
            font=("SF Pro Text", 13, "bold") if platform.system() == "Darwin"
            else ("Segoe UI", 11, "bold"),
            bg=self.accent_color,
            fg="#ffffff",
            activebackground="#0056b3",
            activeforeground="#ffffff",
            relief='flat',
            cursor='hand2',
            width=16,
            height=2,
            command=self.on_start
        )
        start_btn.pack(side='right')

        # Keyboard hint
        hint_label = tk.Label(
            main_frame,
            text="Press Enter to start • Esc to cancel",
            font=("SF Pro Text", 10) if platform.system() == "Darwin"
            else ("Segoe UI", 9),
            fg="#636366",
            bg=self.bg_color
        )
        hint_label.pack(pady=(20, 0))

        # Key bindings
        self.root.bind('<Return>', lambda e: self.on_start())
        self.root.bind('<Escape>', lambda e: self.on_cancel())

    def on_start(self):
        self.should_start = True
        self.root.destroy()

    def on_cancel(self):
        self.should_start = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.should_start


# ============================
# MODERN UI DRAWING FUNCTIONS
# ============================

def draw_rounded_rect(img, pt1, pt2, color, radius=15, thickness=-1, alpha=1.0):
    """Draw a rounded rectangle with optional transparency"""
    x1, y1 = pt1
    x2, y2 = pt2

    if alpha < 1.0:
        overlay = img.copy()
    else:
        overlay = img

    # Draw rounded rectangle using multiple shapes
    # Top-left corner
    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    # Top-right corner
    cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    # Bottom-right corner
    cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    # Bottom-left corner
    cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)

    # Fill rectangles
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_glass_panel(img, pt1, pt2, alpha=0.75, radius=20):
    """Draw a glassmorphism-style panel"""
    x1, y1 = pt1
    x2, y2 = pt2

    # Create overlay for transparency
    overlay = img.copy()

    # Dark background with slight blur effect simulation
    draw_rounded_rect(overlay, pt1, pt2, COLORS_BGR['bg_panel'], radius, -1)

    # Blend with original
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Subtle border
    draw_rounded_rect(img, pt1, pt2, (60, 60, 65), radius, 1)


def draw_progress_ring(img, center, radius, progress, color, thickness=4, bg_color=(60, 60, 65)):
    """Draw a circular progress indicator (Apple Watch style)"""
    cx, cy = center

    # Background ring
    cv2.circle(img, center, radius, bg_color, thickness)

    # Progress arc
    start_angle = -90
    end_angle = start_angle + (progress / 100.0) * 360

    cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, thickness, cv2.LINE_AA)


def draw_mini_chart(img, pt1, pt2, data, color, bg_color=(44, 44, 48)):
    """Draw a mini sparkline chart"""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1

    if len(data) < 2:
        return

    # Normalize data
    min_val = min(data) if data else 0
    max_val = max(data) if data else 100
    range_val = max(max_val - min_val, 1)

    # Draw background
    draw_rounded_rect(img, pt1, pt2, bg_color, 8, -1)

    # Calculate points
    points = []
    for i, val in enumerate(data[-30:]):  # Last 30 points
        px = x1 + 5 + int((i / max(len(data[-30:]) - 1, 1)) * (w - 10))
        py = y2 - 5 - int(((val - min_val) / range_val) * (h - 10))
        points.append((px, py))

    # Draw line
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], color, 2, cv2.LINE_AA)

        # Draw gradient fill below line
        fill_points = points + [(points[-1][0], y2 - 5), (points[0][0], y2 - 5)]
        fill_points = np.array(fill_points, np.int32)

        overlay = img.copy()
        cv2.fillPoly(overlay, [fill_points], color)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)


def draw_status_pill(img, center, status, score):
    """Draw a status indicator pill (iOS-style focus badge)"""

    cx, cy = center

    # Pick color + text
    if status == 'high_focus':
        bg_color = COLORS_BGR['accent_green']
        text = "HIGH FOCUS"
    elif status == 'low_focus':
        bg_color = COLORS_BGR['accent_orange']
        text = "LOW FOCUS"
    else:
        bg_color = COLORS_BGR['accent_red']
        text = "NO FOCUS"

    # Text style
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.70
    thickness = 2

    # Measure text
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Extra padding for clean pill shape
    padding_x = 22
    padding_y = 14

    pill_w = text_w + padding_x * 2
    pill_h = text_h + padding_y * 2

    # Coordinates of pill box
    pt1 = (cx - pill_w // 2, cy - pill_h // 2)
    pt2 = (cx + pill_w // 2, cy + pill_h // 2)

    # Rounded rect radius
    radius = pill_h // 2

    # Draw pill background
    draw_rounded_rect(img, pt1, pt2, bg_color, radius, -1)

    # Center text inside pill
    text_x = cx - text_w // 2
    text_y = cy + text_h // 2 - 2

    cv2.putText(img, text, (text_x, text_y),
                font, font_scale, (255, 255, 255),
                thickness, cv2.LINE_AA)


def draw_metric_card(img, pt1, pt2, icon, label, value, color, show_ring=False, ring_progress=0):
    """Draw a metric card with icon, label and value"""
    x1, y1 = pt1
    x2, y2 = pt2

    # Card background
    draw_rounded_rect(img, pt1, pt2, COLORS_BGR['bg_card'], 12, -1, 0.9)

    # Icon
    cv2.putText(img, icon, (x1 + 12, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    # Label
    cv2.putText(img, label, (x1 + 35, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                COLORS_BGR['text_secondary'], 1, cv2.LINE_AA)

    # Value
    cv2.putText(img, value, (x1 + 12, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                COLORS_BGR['text_primary'], 2, cv2.LINE_AA)

    # Optional progress ring
    if show_ring:
        ring_center = (x2 - 25, (y1 + y2) // 2)
        draw_progress_ring(img, ring_center, 15, ring_progress, color, 3)


def draw_stat_row(img, x, y, label, value, color=None):
    """Draw a single stat row"""
    if color is None:
        color = COLORS_BGR['text_primary']

    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                COLORS_BGR['text_secondary'], 1, cv2.LINE_AA)
    cv2.putText(img, str(value), (x + 100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv2.LINE_AA)


# ============================
# FOCUS TRACKER MAIN CODE
# ============================

mp_face_mesh = mp.solutions.face_mesh


def run_focus_tracker():
    """Main focus tracker function"""

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Camera could not be opened!")
        messagebox.showerror("Error", "Camera could not be opened!\nPlease check your camera connection.")
        return

    WINDOW_NAME = 'Focus Tracker'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    downloads_folder = get_downloads_folder()

    print("=" * 60)
    print("  FOCUS TRACKER v4.0 - Modern UI")
    print("=" * 60)
    print(f"  📁 Reports: {downloads_folder}")
    print("  ⌨️  Press 'Q' to quit and save report")
    print("=" * 60 + "\n")

    # Time variables
    total_high_focus_time = 0
    total_low_focus_time = 0
    total_away_time = 0
    session_start_time = time.time()

    # Smoothing variables
    SMOOTHING_WINDOW = 15
    STATE_CHANGE_THRESHOLD = 0.75

    # Metric weights
    GAZE_WEIGHT = 0.48
    HEAD_WEIGHT = 0.05
    EYE_WEIGHT = 0.47

    # Metric storage
    focus_scores = deque(maxlen=SMOOTHING_WINDOW)
    gaze_scores = deque(maxlen=SMOOTHING_WINDOW)
    head_scores = deque(maxlen=SMOOTHING_WINDOW)
    state_buffer = deque(maxlen=SMOOTHING_WINDOW)

    # Time-series data for charts
    score_history = deque(maxlen=100)

    # Time-series data for report
    timeline = []
    score_timeline = []
    state_timeline = []
    gaze_timeline = []
    head_timeline = []

    # Counters
    blink_count = 0

    # Liveness detection (fotoğraf tespiti)
    LIVENESS_TIMEOUT = 25.0  # 5 saniye hareketsizlik = uyarı
    MOVEMENT_THRESHOLD = 0.008  # Minimum hareket eşiği
    last_iris_positions = None
    last_movement_time = time.time()
    is_live = True
    yawn_count = 0
    distraction_events = 0
    phone_check_count = 0
    downward_look_count = 0

    # State tracking
    last_stable_state = None
    last_blink_time = 0
    eyes_closed_start = None

    # Yawn debounce
    last_yawn_time = 0
    YAWN_COOLDOWN = 1.5

    # Landmarks
    LEFT_EYE_IRIS = [474, 475, 476, 477]
    RIGHT_EYE_IRIS = [469, 470, 471, 472]
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    # Calibration
    CALIB_SECONDS = 2.0
    calib_yaw = []
    calib_pitch = []
    calib_roll = []
    neutral_yaw = neutral_pitch = neutral_roll = 0.0
    calibrated = False

    last_time = time.time()
    frame_count = 0

    # Helper functions (same as before)
    def calculate_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def calculate_eye_aspect_ratio(eye_landmarks):
        A = calculate_distance(eye_landmarks[1], eye_landmarks[5])
        B = calculate_distance(eye_landmarks[2], eye_landmarks[4])
        C = calculate_distance(eye_landmarks[0], eye_landmarks[3])
        return (A + B) / (2.0 * C)

    def normalize_angle(angle):
        return (angle + 180.0) % 360.0 - 180.0

    def calculate_mouth_aspect_ratio(mouth_landmarks):
        vertical = calculate_distance(mouth_landmarks[0], mouth_landmarks[1])
        horizontal = calculate_distance(mouth_landmarks[2], mouth_landmarks[3])
        return vertical / horizontal

    def get_head_pose_angles(landmarks, frame_width, frame_height):
        model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])

        image_points = np.array([
            (landmarks[NOSE_TIP].x * frame_width, landmarks[NOSE_TIP].y * frame_height),
            (landmarks[CHIN].x * frame_width, landmarks[CHIN].y * frame_height),
            (landmarks[LEFT_EYE_OUTER].x * frame_width, landmarks[LEFT_EYE_OUTER].y * frame_height),
            (landmarks[RIGHT_EYE_OUTER].x * frame_width, landmarks[RIGHT_EYE_OUTER].y * frame_height),
            (landmarks[MOUTH_LEFT].x * frame_width, landmarks[MOUTH_LEFT].y * frame_height),
            (landmarks[MOUTH_RIGHT].x * frame_width, landmarks[MOUTH_RIGHT].y * frame_height)
        ], dtype="double")

        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))
        success, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        R, _ = cv2.Rodrigues(rvec)
        P = np.hstack((R, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(P)

        return normalize_angle(float(euler[1])), normalize_angle(float(euler[0])), normalize_angle(float(euler[2]))

    def calculate_gaze_score(face_landmarks, iris_ids, eye_ids, frame_width, frame_height):
        iris_pts = np.array([[face_landmarks[i].x * frame_width, face_landmarks[i].y * frame_height] for i in iris_ids])
        iris_cx, iris_cy = iris_pts.mean(axis=0)

        eye_pts = np.array([[face_landmarks[i].x * frame_width, face_landmarks[i].y * frame_height] for i in eye_ids])
        min_x, min_y = eye_pts.min(axis=0)
        max_x, max_y = eye_pts.max(axis=0)

        eye_w = max(max_x - min_x, 1e-6)
        eye_h = max(max_y - min_y, 1e-6)

        nx = (iris_cx - min_x) / eye_w
        ny = (iris_cy - min_y) / eye_h

        dx = abs(nx - 0.5) / 0.5
        dy = abs(ny - 0.5) / 0.5
        r = math.sqrt(dx * dx + dy * dy)

        r0, r1 = 0.30, 1.20
        if r <= r0:
            score = 100.0
        elif r >= r1:
            score = 0.0
        else:
            score = 100.0 * (1 - (r - r0) / (r1 - r0))

        return float(np.clip(score, 0, 100)), dx, dy

    def calculate_head_score(yaw, pitch, roll):
        # Her açıyı ayrı ayrı değerlendir
        yaw_penalty = min(abs(yaw) / 45.0, 1.0) * 100  # 45 derece = max penalty
        pitch_penalty = min(abs(pitch) / 35.0, 1.0) * 100  # 35 derece = max penalty
        roll_penalty = min(abs(roll) / 30.0, 1.0) * 50  # Roll daha az önemli

        # En kötü açıyı baz al
        max_penalty = max(yaw_penalty, pitch_penalty, roll_penalty)

        score = 100.0 - max_penalty
        return float(np.clip(score, 0, 100))

    def analyze_focus(gaze_left, gaze_right, head_score, ear_avg, yaw, pitch, roll):
        warnings = []
        gaze_avg = (gaze_left + gaze_right) / 2

        # Eye open score
        if ear_avg <= 0.12:
            eye_open_score = 0.0
        elif ear_avg >= 0.28:
            eye_open_score = 100.0
        else:
            eye_open_score = (ear_avg - 0.12) / (0.28 - 0.12) * 100.0
        eye_open_score = float(np.clip(eye_open_score, 0, 100))

        # Weighting (boosting kaldırıldı - gerçek değerler kullanılıyor)
        total_score = gaze_avg * GAZE_WEIGHT + head_score * HEAD_WEIGHT + eye_open_score * EYE_WEIGHT

        # Warnings
        if abs(yaw) > 50:
            warnings.append("Head turned sideways")
        if pitch < -45:
            warnings.append("Looking down")
        elif pitch > 15:
            warnings.append("Looking up")
        if gaze_avg < 12:
            warnings.append("Eyes off screen")
        if ear_avg < 0.13:
            warnings.append("Eyes closing")

        # --- Pitch rule ---
        forced_no_focus = False
        if pitch < -45:
            forced_no_focus = True
        elif pitch < -35 and gaze_avg < 40:
            forced_no_focus = True
        elif pitch > 15:
            forced_no_focus = True
        elif pitch > 8 and gaze_avg < 60:
            forced_no_focus = True

        # --- Hysteresis thresholds ---
        HIGH_ON = 80
        HIGH_OFF = 72
        LOW_ON = 30
        LOW_OFF = 22

        if forced_no_focus:
            state = 'no_focus'
        else:
            prev = last_stable_state

            if prev == 'high_focus':
                if total_score >= HIGH_OFF:
                    state = 'high_focus'
                elif total_score >= LOW_ON:
                    state = 'low_focus'
                else:
                    state = 'no_focus'

            elif prev == 'low_focus':
                if total_score >= HIGH_ON:
                    state = 'high_focus'
                elif total_score >= LOW_OFF:
                    state = 'low_focus'
                else:
                    state = 'no_focus'

            else:
                if gaze_avg >= 85 and eye_open_score >= 70 and total_score >= 70:
                    total_score = max(total_score, 80)

                if total_score >= HIGH_ON:
                    state = 'high_focus'
                elif total_score >= LOW_ON:
                    state = 'low_focus'
                else:
                    state = 'no_focus'

        return state, float(np.clip(total_score, 0, 100)), {
            'gaze': gaze_avg,
            'head': head_score,  # Gerçek head_score döndürülüyor
            'eyes': eye_open_score
        }, warnings

    def smooth_state(current_state, state_buffer):
        if len(state_buffer) < SMOOTHING_WINDOW * 0.5:
            return current_state

        state_counts = {'high_focus': 0, 'low_focus': 0, 'no_focus': 0, 'away': 0}
        for s in state_buffer:
            if s in state_counts:
                state_counts[s] += 1

        max_count = max(state_counts.values())
        max_state = max(state_counts, key=state_counts.get)

        if max_count / len(state_buffer) >= STATE_CHANGE_THRESHOLD:
            return max_state
        return state_buffer[-2] if len(state_buffer) >= 2 else current_state

    def format_time(seconds):
        return str(timedelta(seconds=int(seconds)))

    def compute_final_score(avg_score, high_pct, low_pct, distractions, phone_checks, yawns, session_minutes):
        # Dakika başına normalize et
        time_factor = max(session_minutes / 5.0, 1.0)

        normalized_distractions = distractions / time_factor
        normalized_yawns = yawns / time_factor

        penalty = 2.0 * normalized_distractions + 1.0 * normalized_yawns
        penalty = min(penalty, 15)

        # Basit formül: avg_score ağırlıklı, high_pct bonus
        raw = avg_score * 0.85 + high_pct * 0.15 - penalty
        return float(np.clip(raw, 0, 100))

    def letter_grade(score):
        if score >= 90: return "A+"
        if score >= 80: return "A"
        if score >= 70: return "B"
        if score >= 60: return "C"
        if score >= 50: return "D"
        return "F"

    def generate_modern_report(timeline, score_timeline, state_timeline, gaze_timeline,
                               head_timeline, total_high, total_low, total_away,
                               session_stats, downloads_folder):
        """Generate a cleaner, more modern LIGHT report with pill legend on state timeline."""

        filename_base = get_report_filename()
        png_path = os.path.join(downloads_folder, f"{filename_base}.png")
        pdf_path = os.path.join(downloads_folder, f"{filename_base}.pdf")

        # ---- LIGHT THEME BASE ----
        plt.rcParams.update({
            "font.family": "sans-serif",
            "axes.edgecolor": "#E5E5EA",
            "axes.labelcolor": "#1C1C1E",
            "xtick.color": "#3A3A3C",
            "ytick.color": "#3A3A3C",
            "grid.color": "#ECECF1"
        })

        # Palette
        blue = "#007AFF"
        green = "#34C759"
        orange = "#FF9500"
        red = "#FF3B30"
        gray = "#8E8E93"
        dark = "#1C1C1E"
        mid = "#3A3A3C"
        card = "#F2F2F7"
        white = "#FFFFFF"

        fig = plt.figure(figsize=(16, 10), facecolor=white)
        fig.patch.set_facecolor(white)

        # Title
        fig.suptitle("Focus Session Report", fontsize=26, fontweight="bold",
                     color=dark, y=0.97)
        subtitle = (
            f"{datetime.now().strftime('%B %d, %Y at %H:%M')}  •  "
            f"Duration: {session_stats['total_time']}  •  "
            f"Grade: {session_stats['grade']}"
        )
        fig.text(0.5, 0.92, subtitle, ha="center", fontsize=11, color=gray)

        # Layout:
        # Row1: [Score Card big] [Session Stats big]
        # Row2: [Score Timeline full width]
        # Row3: [Focus Distribution bar] [Submetrics]
        # Row4: [State Timeline full width]
        gs = fig.add_gridspec(
            4, 6,
            height_ratios=[1.15, 1.4, 1.05, 0.9],
            hspace=0.45, wspace=0.35,
            left=0.05, right=0.95, top=0.88, bottom=0.06
        )

        # ---------------------------
        # 1) FINAL SCORE CARD (modern, no ring)
        # ---------------------------
        ax_score = fig.add_subplot(gs[0, :3])
        ax_score.set_facecolor(card)
        ax_score.set_xlim(0, 1)
        ax_score.set_ylim(0, 1)
        ax_score.axis("off")

        final_score = session_stats["final_score"]
        avg_score = session_stats["avg_score"]

        ax_score.text(0.06, 0.62, f"{final_score:.0f}",
                      fontsize=52, fontweight="bold", color=dark, va="center")

        ax_score.text(0.06, 0.36, "Final Focus Score",
                      fontsize=12, color=gray, va="center")

        # Progress bar
        bar_x, bar_y, bar_w, bar_h = 0.06, 0.16, 0.88, 0.08
        ax_score.add_patch(plt.Rectangle((bar_x, bar_y), bar_w, bar_h,
                                         color=white, ec="#E5E5EA", lw=1))
        ax_score.add_patch(plt.Rectangle((bar_x, bar_y),
                                         bar_w * (final_score / 100),
                                         bar_h,
                                         color=blue, lw=0))

        ax_score.text(0.06, 0.05,
                      f"Avg score: {avg_score:.1f}%   •   High focus: {session_stats['high_pct']:.1f}%",
                      fontsize=10, color=mid)

        for sp in ax_score.spines.values():
            sp.set_visible(False)

        # ---------------------------
        # 2) SESSION STATS BIG (top-right)
        # ---------------------------
        ax_stats = fig.add_subplot(gs[0, 3:])
        ax_stats.set_facecolor(card)
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)
        ax_stats.axis("off")

        ax_stats.text(0.06, 0.86, "Session Stats",
                      fontsize=14, fontweight="bold", color=dark)

        stats_left = [
            ("Session Duration", session_stats["total_time"]),
            ("Average Score", f"{session_stats['avg_score']:.1f}%"),
            ("High Focus", f"{session_stats['high_pct']:.1f}%"),
            ("Low Focus", f"{session_stats['low_pct']:.1f}%"),
            ("Away/None", f"{session_stats['away_pct']:.1f}%"),
        ]
        stats_right = [
            ("Distractions", session_stats["distractions"]),
            ("Yawns", session_stats["yawns"]),
        ]
        # Sol sütun: label x=0.06, value x=0.46  (ÇAKIŞMAYI BİTİREN NOKTA)
        y = 0.70
        for label, value in stats_left:
            ax_stats.text(0.06, y, label, fontsize=11, color=gray, va="center")
            ax_stats.text(0.46, y, str(value), fontsize=11,
                          color=dark, va="center", ha="right", fontweight="bold")
            y -= 0.12

        # Sağ sütun: label x=0.56, value x=0.94 (aynı kalsın)
        y = 0.70
        for label, value in stats_right:
            ax_stats.text(0.56, y, label, fontsize=11, color=gray, va="center")
            ax_stats.text(0.94, y, str(value), fontsize=11,
                          color=dark, va="center", ha="right", fontweight="bold")
            y -= 0.12

        for sp in ax_stats.spines.values():
            sp.set_visible(False)

        # ---------------------------
        # 3) FOCUS SCORE OVER TIME (full width)
        # ---------------------------
        ax_timeline = fig.add_subplot(gs[1, :])
        ax_timeline.set_facecolor(card)

        if timeline and score_timeline:
            time_minutes = [t / 60 for t in timeline]
            ax_timeline.fill_between(time_minutes, score_timeline, alpha=0.12, color=blue)
            ax_timeline.plot(time_minutes, score_timeline, color=blue, lw=2)
            ax_timeline.axhline(y=80, ls="--", alpha=0.45, color=green, lw=1)
            ax_timeline.axhline(y=30, ls="--", alpha=0.45, color=orange, lw=1)

        ax_timeline.set_title("Focus Score Over Time", fontsize=14,
                              fontweight="bold", color=dark, pad=8)
        ax_timeline.set_xlabel("Time (minutes)", fontsize=10, color=mid)
        ax_timeline.set_ylabel("Focus Score", fontsize=10, color=mid)
        ax_timeline.set_ylim(0, 100)
        ax_timeline.grid(True, alpha=0.7)
        for sp in ax_timeline.spines.values():
            sp.set_color("#E5E5EA")

        # ---------------------------
        # 4) FOCUS DISTRIBUTION as THIN HORIZONTAL SEGMENT BAR
        # ---------------------------
        ax_dist = fig.add_subplot(gs[2, :3])
        ax_dist.set_facecolor(card)
        ax_dist.set_xlim(0, 100)
        ax_dist.set_ylim(0, 1)
        ax_dist.axis("off")

        ax_dist.text(0.06, 0.85, "Focus Distribution",
                     transform=ax_dist.transAxes,
                     fontsize=13, fontweight="bold", color=dark)

        total_time = total_high + total_low + total_away
        if total_time > 0:
            high_pct = (total_high / total_time) * 100
            low_pct = (total_low / total_time) * 100
            away_pct = (total_away / total_time) * 100

            bar_y = 0.45
            bar_h = 0.18

            start = 0
            ax_dist.add_patch(plt.Rectangle((start, bar_y), high_pct, bar_h, color=green, lw=0))
            start += high_pct
            ax_dist.add_patch(plt.Rectangle((start, bar_y), low_pct, bar_h, color=orange, lw=0))
            start += low_pct
            ax_dist.add_patch(plt.Rectangle((start, bar_y), away_pct, bar_h, color=red, lw=0))

            ax_dist.text(0, 0.20, f"High Focus  {high_pct:.1f}%", color=green, fontsize=10, fontweight="bold")
            ax_dist.text(38, 0.20, f"Low Focus  {low_pct:.1f}%", color=orange, fontsize=10, fontweight="bold")
            ax_dist.text(68, 0.20, f"Away/None  {away_pct:.1f}%", color=red, fontsize=10, fontweight="bold")

        for sp in ax_dist.spines.values():
            sp.set_visible(False)

        # ---------------------------
        # 5) SUB-METRICS OVER TIME (right)
        # ---------------------------
        ax_metrics = fig.add_subplot(gs[2, 3:])
        ax_metrics.set_facecolor(card)

        if timeline:
            time_minutes = [t / 60 for t in timeline]
            ax_metrics.plot(time_minutes, gaze_timeline, color="#5AC8FA", lw=2, label="Gaze")
            ax_metrics.plot(time_minutes, head_timeline, color=orange, lw=2, label="Head Pose")

        ax_metrics.set_title("Sub-metrics Over Time", fontsize=13,
                             fontweight="bold", color=dark, pad=8)
        ax_metrics.set_xlabel("Time (minutes)", fontsize=10, color=mid)
        ax_metrics.set_ylabel("Score", fontsize=10, color=mid)
        ax_metrics.set_ylim(0, 100)
        ax_metrics.grid(True, alpha=0.7)
        ax_metrics.legend(loc="lower right", frameon=False, labelcolor=dark)
        for sp in ax_metrics.spines.values():
            sp.set_color("#E5E5EA")

        # ---------------------------
        # 6) STATE TIMELINE (full width) + PILL LEGEND
        # ---------------------------
        ax_state = fig.add_subplot(gs[3, :])
        ax_state.set_facecolor(card)

        if timeline and state_timeline:
            time_minutes = [t / 60 for t in timeline]
            colors_map = {"high_focus": green, "low_focus": orange, "no_focus": red, "away": red}

            for i in range(len(time_minutes) - 1):
                st = state_timeline[i]
                ax_state.axvspan(time_minutes[i], time_minutes[i + 1],
                                 facecolor=colors_map.get(st, red), alpha=0.85)

        ax_state.set_title("Focus State Timeline", fontsize=13,
                           fontweight="bold", color=dark, pad=6)
        ax_state.set_xlabel("Time (minutes)", fontsize=10, color=mid)
        ax_state.set_yticks([])
        for sp in ax_state.spines.values():
            sp.set_color("#E5E5EA")

        # --- PILL LEGEND (Apple style) ---
        # Place 3 small pills inside axis, top-right, no clunky box
        pill_kwargs = dict(
            transform=ax_state.transAxes,
            fontsize=9.5,
            color="white",
            va="center",
            ha="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8", ec="none")
        )

        ax_state.text(0.80, 0.85, "High Focus",
                      **{**pill_kwargs, "bbox": {**pill_kwargs["bbox"], "fc": green}})
        ax_state.text(0.90, 0.85, "Low Focus",
                      **{**pill_kwargs, "bbox": {**pill_kwargs["bbox"], "fc": orange}})
        ax_state.text(0.98, 0.85, "Away/None",
                      **{**pill_kwargs, "bbox": {**pill_kwargs["bbox"], "fc": red}})

        # Save outputs
        plt.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=white)
        print(f"✓ PNG saved: {png_path}")

        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, bbox_inches="tight", facecolor=white)
        print(f"✓ PDF saved: {pdf_path}")

        return png_path, pdf_path

    # Main loop
    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True,
            max_num_faces=1
    ) as face_mesh:

        print("✓ Camera ready\n")

        while True:
            ret, frame_raw = cap.read()
            if not ret:
                break

            current_time = time.time()
            elapsed_time = current_time - last_time
            elapsed_from_start = current_time - session_start_time

            frame_raw = cv2.flip(frame_raw, 1)
            h0, w0 = frame_raw.shape[:2]
            rgb_raw = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_raw)

            # Resize for display
            frame = frame_raw.copy()
            (_, _, screen_w, screen_h) = cv2.getWindowImageRect(WINDOW_NAME)
            if screen_w > 0 and screen_h > 0:
                frame = cv2.resize(frame, (screen_w, screen_h))
            h, w = frame.shape[:2]
            sx, sy = w / w0, h / h0

            current_state = 'away'
            total_score = 0
            sub_scores = {'gaze': 0, 'head': 0, 'eyes': 0}
            warnings = []
            yaw_c = pitch_c = roll_c = 0

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark

                # Eye aspect ratio
                left_eye_coords = np.array([[face_landmarks[i].x * w0, face_landmarks[i].y * h0] for i in LEFT_EYE])
                right_eye_coords = np.array([[face_landmarks[i].x * w0, face_landmarks[i].y * h0] for i in RIGHT_EYE])
                ear_left = calculate_eye_aspect_ratio(left_eye_coords)
                ear_right = calculate_eye_aspect_ratio(right_eye_coords)
                ear_avg = (ear_left + ear_right) / 2

                # Blink detection
                if ear_avg < 0.18:
                    if eyes_closed_start is None:
                        eyes_closed_start = current_time
                    elif current_time - eyes_closed_start > 0.3:
                        if current_time - last_blink_time > 2:
                            blink_count += 1
                            last_blink_time = current_time
                else:
                    eyes_closed_start = None

                # Yawn detection
                mouth_landmarks = [
                    (face_landmarks[MOUTH_TOP].x * w0, face_landmarks[MOUTH_TOP].y * h0),
                    (face_landmarks[MOUTH_BOTTOM].x * w0, face_landmarks[MOUTH_BOTTOM].y * h0),
                    (face_landmarks[MOUTH_LEFT].x * w0, face_landmarks[MOUTH_LEFT].y * h0),
                    (face_landmarks[MOUTH_RIGHT].x * w0, face_landmarks[MOUTH_RIGHT].y * h0)
                ]
                mar = calculate_mouth_aspect_ratio(mouth_landmarks)
                if mar > 0.6 and (current_time - last_yawn_time) > YAWN_COOLDOWN:
                    yawn_count += 1
                    last_yawn_time = current_time

                # Head pose
                yaw, pitch, roll = get_head_pose_angles(face_landmarks, w0, h0)

                if not calibrated:
                    # Sadece stabil duruşları kalibre et
                    # (başlangıçta kafayı oynatırken yanlış baseline almaması için)
                    if abs(yaw) < 25 and abs(pitch) < 25 and abs(roll) < 25 and ear_avg > 0.15:
                        calib_yaw.append(yaw)
                        calib_pitch.append(pitch)
                        calib_roll.append(roll)

                    head_score = 100.0

                    if elapsed_from_start >= CALIB_SECONDS and len(calib_pitch) > 10:
                        neutral_yaw = float(np.median(calib_yaw))
                        neutral_pitch = float(np.median(calib_pitch))
                        neutral_roll = float(np.median(calib_roll))
                        calibrated = True
                        print("✓ Calibration complete")

                    # Kalibrasyon bitene kadar corrected açıları 0 say
                    yaw_c, pitch_c, roll_c = 0.0, 0.0, 0.0

                else:
                    yaw_c = yaw - neutral_yaw
                    pitch_c = pitch - neutral_pitch
                    roll_c = roll - neutral_roll
                    head_score = calculate_head_score(yaw_c, pitch_c, roll_c)

                # Phone check
                if pitch_c < -35:
                    downward_look_count += 1
                    if downward_look_count % 30 == 0:
                        phone_check_count += 1
                else:
                    downward_look_count = 0

                    # Gaze
                    gaze_left, _, _ = calculate_gaze_score(face_landmarks, LEFT_EYE_IRIS, LEFT_EYE, w0, h0)
                    gaze_right, _, _ = calculate_gaze_score(face_landmarks, RIGHT_EYE_IRIS, RIGHT_EYE, w0, h0)

                    # Liveness detection - iris hareketini kontrol et
                    current_iris_positions = []
                    for iris_id in LEFT_EYE_IRIS + RIGHT_EYE_IRIS:
                        current_iris_positions.append((face_landmarks[iris_id].x, face_landmarks[iris_id].y))

                    if last_iris_positions is not None:
                        # Ortalama hareket miktarını hesapla
                        total_movement = 0
                        for i, (curr, prev) in enumerate(zip(current_iris_positions, last_iris_positions)):
                            movement = math.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
                            total_movement += movement
                        avg_movement = total_movement / len(current_iris_positions)

                        # Hareket varsa zamanı güncelle
                        if avg_movement > MOVEMENT_THRESHOLD:
                            last_movement_time = current_time
                            is_live = True
                        else:
                            # Hareketsizlik süresi kontrolü
                            if current_time - last_movement_time > LIVENESS_TIMEOUT:
                                is_live = False

                    last_iris_positions = current_iris_positions

                    # Focus analysis
                current_state, total_score, sub_scores, warnings = analyze_focus(
                    gaze_left, gaze_right, head_score, ear_avg, yaw_c, pitch_c, roll_c
                )

                # Draw iris markers (subtle)
                for iris_id in LEFT_EYE_IRIS + RIGHT_EYE_IRIS:
                    iris = face_landmarks[iris_id]
                    cx, cy = int(iris.x * w0 * sx), int(iris.y * h0 * sy)
                    cv2.circle(frame, (cx, cy), 2, COLORS_BGR['accent_cyan'], -1, cv2.LINE_AA)

            # Update buffers
            focus_scores.append(total_score)
            state_buffer.append(current_state)
            score_history.append(total_score)

            smoothed_state = smooth_state(current_state, state_buffer)

            # Timeline data
            if frame_count % 15 == 0:
                timeline.append(elapsed_from_start)
                score_timeline.append(total_score)
                state_timeline.append(smoothed_state)
                gaze_timeline.append(sub_scores['gaze'])
                head_timeline.append(sub_scores['head'])

            # Time accumulation
            if smoothed_state == 'high_focus':
                total_high_focus_time += elapsed_time
            elif smoothed_state == 'low_focus':
                total_low_focus_time += elapsed_time
            else:
                total_away_time += elapsed_time

            # Distraction detection
            if last_stable_state and last_stable_state != smoothed_state:
                if smoothed_state in ['low_focus', 'no_focus', 'away'] and last_stable_state == 'high_focus':
                    distraction_events += 1

            last_stable_state = smoothed_state

            # =====================
            # MODERN UI RENDERING
            # =====================

            ui_scale = min(max(min(w / 1280.0, h / 720.0), 0.6), 1.4)
            margin = int(16 * ui_scale)

            # Left panel - Main stats
            panel_w = int(280 * ui_scale)
            panel_h = int(320 * ui_scale)
            draw_glass_panel(frame, (margin, margin), (margin + panel_w, margin + panel_h), 0.85, 16)

            # Status pill at top of panel
            status_y = margin + int(35 * ui_scale)
            draw_status_pill(frame, (margin + panel_w // 2, status_y), smoothed_state, total_score)

            # Score display
            score_y = margin + int(85 * ui_scale)
            cv2.putText(frame, f"{int(total_score)}", (margin + int(20 * ui_scale), score_y + int(50 * ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2 * ui_scale, COLORS_BGR['text_primary'], 3, cv2.LINE_AA)
            cv2.putText(frame, "%", (margin + int(100 * ui_scale), score_y + int(25 * ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * ui_scale, COLORS_BGR['text_secondary'], 1, cv2.LINE_AA)

            # Mini chart
            chart_y = score_y + int(70 * ui_scale)
            draw_mini_chart(frame,
                            (margin + int(15 * ui_scale), chart_y),
                            (margin + panel_w - int(15 * ui_scale), chart_y + int(50 * ui_scale)),
                            list(score_history), COLORS_BGR['accent_blue'])

            # Sub-metrics
            metrics_y = chart_y + int(70 * ui_scale)
            line_height = int(28 * ui_scale)

            metrics = [
                ("Gaze", f"{int(sub_scores['gaze'])}%", COLORS_BGR['accent_cyan']),
                ("Head Pose", f"{int(sub_scores['head'])}%", COLORS_BGR['accent_orange']),
                ("Eyes", f"{int(sub_scores['eyes'])}%", COLORS_BGR['accent_purple']),
            ]

            for i, (label, value, color) in enumerate(metrics):
                y = metrics_y + i * line_height
                cv2.putText(frame, label, (margin + int(20 * ui_scale), y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * ui_scale, COLORS_BGR['text_secondary'], 1, cv2.LINE_AA)
                cv2.putText(frame, value, (margin + panel_w - int(50 * ui_scale), y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ui_scale, color, 1, cv2.LINE_AA)

                # Progress bar
                bar_x = margin + int(90 * ui_scale)
                bar_w = int(100 * ui_scale)
                bar_h = int(4 * ui_scale)
                progress = int(
                    sub_scores[label.lower().replace(' ', '_').replace('_pose', '')] if label != "Eyes" else sub_scores[
                        'eyes'])

                # Background bar
                cv2.rectangle(frame, (bar_x, y - int(8 * ui_scale)),
                              (bar_x + bar_w, y - int(8 * ui_scale) + bar_h),
                              (60, 60, 65), -1)
                # Progress bar
                cv2.rectangle(frame, (bar_x, y - int(8 * ui_scale)),
                              (bar_x + int(bar_w * progress / 100), y - int(8 * ui_scale) + bar_h),
                              color, -1)

            # Right panel - Session stats
            right_panel_w = int(180 * ui_scale)
            right_panel_h = int(200 * ui_scale)
            right_x = w - margin - right_panel_w
            draw_glass_panel(frame, (right_x, margin), (w - margin, margin + right_panel_h), 0.85, 16)

            # Session title
            cv2.putText(frame, "SESSION", (right_x + int(15 * ui_scale), margin + int(28 * ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * ui_scale, COLORS_BGR['text_secondary'], 1, cv2.LINE_AA)

            # Time display
            session_time = int(current_time - session_start_time)
            mins, secs = divmod(session_time, 60)
            time_str = f"{mins:02d}:{secs:02d}"
            cv2.putText(frame, time_str, (right_x + int(15 * ui_scale), margin + int(60 * ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * ui_scale, COLORS_BGR['text_primary'], 2, cv2.LINE_AA)

            # Stats
            stats_y = margin + int(90 * ui_scale)
            stats = [
                ("High", f"{int(total_high_focus_time)}s", COLORS_BGR['accent_green']),
                ("Low", f"{int(total_low_focus_time)}s", COLORS_BGR['accent_orange']),
                ("Away", f"{int(total_away_time)}s", COLORS_BGR['accent_red']),
                ("Distractions", str(distraction_events), COLORS_BGR['text_secondary']),
            ]

            for i, (label, value, color) in enumerate(stats):
                y = stats_y + i * int(26 * ui_scale)
                cv2.putText(frame, label, (right_x + int(15 * ui_scale), y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ui_scale, COLORS_BGR['text_tertiary'], 1, cv2.LINE_AA)
                cv2.putText(frame, value, (right_x + right_panel_w - int(35 * ui_scale), y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * ui_scale, color, 1, cv2.LINE_AA)
                # Liveness warning (fotoğraf uyarısı)
                if not is_live and results.multi_face_landmarks:
                    warn_text = "NO MOVEMENT DETECTED - Please move your eyes"
                    (tw, th), _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui_scale, 2)
                    liveness_warn_x = (w - tw) // 2
                    liveness_warn_y = h // 2

                    # Kırmızı uyarı kutusu
                    cv2.rectangle(frame, (liveness_warn_x - 20, liveness_warn_y - th - 20),
                                  (liveness_warn_x + tw + 20, liveness_warn_y + 20),
                                  COLORS_BGR['accent_red'], -1)
                    cv2.putText(frame, warn_text, (liveness_warn_x, liveness_warn_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui_scale, (255, 255, 255), 2, cv2.LINE_AA)

                    # Hareketsizken focus'u sıfırla
                    current_state = 'no_focus'
                    total_score = 0

                # Warnings (bottom left, subtle)
                if warnings and results.multi_face_landmarks and is_live:
                    warnings_base_y = h - margin - int(30 * ui_scale * len(warnings[:2]))
                    for i, warning in enumerate(warnings[:2]):
                        y = warnings_base_y + i * int(25 * ui_scale)
                    # Warning pill
                    (tw, th), _ = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ui_scale, 1)
                    cv2.rectangle(frame, (margin, y - int(15 * ui_scale)),
                                  (margin + tw + int(20 * ui_scale), y + int(8 * ui_scale)),
                                  (40, 40, 45), -1)
                    cv2.putText(frame, f"⚠ {warning}", (margin + int(8 * ui_scale), y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ui_scale, COLORS_BGR['accent_orange'], 1, cv2.LINE_AA)

            # Bottom center hint
            hint_text = "Press Q to quit and save report"
            (hint_w, hint_h), _ = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * ui_scale, 1)
            hint_x = (w - hint_w) // 2
            hint_y = h - margin

            # Hint background
            cv2.rectangle(frame, (hint_x - int(15 * ui_scale), hint_y - hint_h - int(10 * ui_scale)),
                          (hint_x + hint_w + int(15 * ui_scale), hint_y + int(5 * ui_scale)),
                          (30, 30, 35), -1)
            cv2.putText(frame, hint_text, (hint_x, hint_y - int(3 * ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * ui_scale, COLORS_BGR['text_tertiary'], 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)

            last_time = current_time
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Generate report
    print("\n" + "=" * 60)
    print("  Generating Report...")
    print("=" * 60)

    final_session_time = time.time() - session_start_time

    high_pct = low_pct = away_pct = 0
    if final_session_time > 0:
        high_pct = (total_high_focus_time / final_session_time) * 100
        low_pct = (total_low_focus_time / final_session_time) * 100
        away_pct = (total_away_time / final_session_time) * 100

    avg_focus = np.mean(score_timeline) if score_timeline else 0
    session_minutes = final_session_time / 60.0
    session_minutes = final_session_time / 60.0
    final_score = compute_final_score(avg_focus, high_pct, low_pct, distraction_events, phone_check_count, yawn_count,
                                      session_minutes)
    grade = letter_grade(final_score)
    session_stats = {
        'total_time': format_time(final_session_time),
        'high_focus_time': format_time(total_high_focus_time),
        'low_focus_time': format_time(total_low_focus_time),
        'away_time': format_time(total_away_time),
        'high_pct': high_pct,
        'low_pct': low_pct,
        'away_pct': away_pct,
        'avg_score': avg_focus,
        'final_score': final_score,
        'grade': grade,
        'distractions': distraction_events,
        'phone_checks': phone_check_count,
        'blinks': blink_count,
        'yawns': yawn_count
    }

    png_path, pdf_path = generate_modern_report(
        timeline, score_timeline, state_timeline,
        gaze_timeline, head_timeline,
        total_high_focus_time, total_low_focus_time, total_away_time,
        session_stats, downloads_folder
    )

    print("\n" + "=" * 60)
    print("  ✓ Session Complete!")
    print("=" * 60)
    print(f"  📊 Final Score: {final_score:.1f}/100 ({grade})")
    print(f"  📁 Saved to Downloads:")
    print(f"     • {os.path.basename(png_path)}")
    print(f"     • {os.path.basename(pdf_path)}")
    print("=" * 60)

    plt.show()

    messagebox.showinfo(
        "Session Complete",
        f"Focus session completed!\n\n"
        f"Final Score: {final_score:.1f}/100 ({grade})\n\n"
        f"Reports saved to Downloads folder."
    )


# ============================
# MAIN ENTRY POINT
# ============================

if __name__ == "__main__":
    dialog = ModernStartDialog()
    should_start = dialog.run()

    if should_start:
        run_focus_tracker()
    else:
        print("Session cancelled.")
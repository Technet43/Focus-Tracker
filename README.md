# 🎯 Focus Tracker v4.0

**AI-Powered Focus Analysis with Real-Time Monitoring**

An intelligent focus tracking application that uses computer vision and MediaPipe to monitor your attention levels during work or study sessions. Features a modern Apple-inspired UI and generates comprehensive session reports.

---

## ✨ Features

### Real-Time Focus Detection
- **Gaze Tracking** — Monitors eye position and iris movement to detect where you're looking
- **Head Pose Estimation** — Analyzes head orientation (yaw, pitch, roll) using 3D face landmarks
- **Eye Aspect Ratio (EAR)** — Detects blinks, eye closure, and drowsiness
- **Liveness Detection** — Identifies if a static image is being used instead of a live face

### Smart Analytics
- **Weighted Focus Score** — Combines gaze (48%), eye openness (47%), and head pose (5%) for accurate assessment
- **State Classification** — Categorizes focus into High Focus, Low Focus, and No Focus states
- **Hysteresis Thresholds** — Prevents flickering between states with intelligent smoothing
- **Distraction & Yawn Detection** — Tracks behavioral events that impact productivity

### Modern UI Design
- Apple-inspired dark glassmorphism interface
- Real-time focus score display with progress ring
- Live mini sparkline chart showing score history
- Color-coded status pills (green/orange/red)
- Metric breakdown panels for gaze, head pose, and eye tracking

### Comprehensive Reports
- **PDF & PNG Export** — Automatically saved to Downloads folder
- **Session Statistics** — Total time, focus percentages, event counts
- **Score Timeline Graph** — Visualizes focus fluctuations over time
- **State Distribution Chart** — Shows time spent in each focus state
- **Final Grade** — A+ to F rating based on overall performance

---

## 🛠 Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
```
opencv-python
mediapipe
numpy
matplotlib
```

### Installation

```bash
# Clone or download the repository
git clone https://github.com/yourusername/focus-tracker.git
cd focus-tracker

# Install dependencies
pip install opencv-python mediapipe numpy matplotlib

# Run the application
python Focus_Tracker_Beta.py
```

---

## 🚀 Usage

### Starting a Session

1. Run the script — a modern start dialog will appear
2. Click **Start Session** or press **Enter**
3. Position yourself in front of the camera
4. The app will calibrate for 2 seconds (stay still and look at center)
5. Begin your work session

### During the Session

| Indicator | Meaning |
|-----------|---------|
| 🟢 **HIGH FOCUS** | Score ≥80% — Excellent attention |
| 🟠 **LOW FOCUS** | Score 30-79% — Partial attention |
| 🔴 **NO FOCUS** | Score <30% — Distracted or away |

### Ending a Session

- Press **Q** to quit and generate reports
- Reports are automatically saved to your Downloads folder
- A summary dialog shows your final score and grade

---

## 📊 How Scoring Works

### Focus Score Calculation

```
Total Score = (Gaze Score × 0.48) + (Eye Score × 0.47) + (Head Score × 0.05)
```

### Gaze Score
- Measures iris position relative to eye boundaries
- 100% when looking directly at screen center
- Decreases as gaze moves toward periphery

### Eye Openness Score
- Based on Eye Aspect Ratio (EAR)
- EAR ≤ 0.12 → 0% (eyes closed)
- EAR ≥ 0.28 → 100% (eyes fully open)
- Linear interpolation between thresholds

### Head Pose Score
- Penalizes head rotation away from neutral position
- Yaw (left/right): Max 45° tolerance
- Pitch (up/down): Max 35° tolerance
- Roll (tilt): Max 30° tolerance

### Final Score Formula

```
Final = (Avg Score × 0.85) + (High Focus % × 0.15) - Penalties
```

Penalties are applied for distractions and yawns, normalized by session duration.

---

## 📁 Output Files

Reports are saved to your system's Downloads folder:

```
FocusTracker_Report_2024-01-15_14-30-22.png
FocusTracker_Report_2024-01-15_14-30-22.pdf
```

### Report Contents

- **Final Focus Score** with progress bar
- **Session Statistics** (duration, averages, percentages)
- **Focus Score Timeline** graph
- **State Distribution** visualization
- **Grade** (A+, A, B, C, D, F)

---

## ⚙️ Configuration

Key parameters can be adjusted in the source code:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SMOOTHING_WINDOW` | 15 | Frames for state smoothing |
| `STATE_CHANGE_THRESHOLD` | 0.75 | Minimum ratio to change state |
| `GAZE_WEIGHT` | 0.48 | Weight of gaze in total score |
| `EYE_WEIGHT` | 0.47 | Weight of eye openness |
| `HEAD_WEIGHT` | 0.05 | Weight of head pose |
| `LIVENESS_TIMEOUT` | 25.0s | Time before "no movement" warning |
| `YAWN_COOLDOWN` | 1.5s | Minimum time between yawn detections |

---

## 🔧 Troubleshooting

### Camera Not Found
```
ERROR: Camera could not be opened!
```
- Check if another application is using the camera
- Verify camera permissions in system settings
- Try unplugging and reconnecting USB webcams

### Low Frame Rate
- Close other resource-intensive applications
- Ensure adequate lighting for face detection
- Consider lowering camera resolution

### Inconsistent Tracking
- Maintain consistent lighting (avoid backlighting)
- Keep face clearly visible to camera
- Stay within 30-80cm of the camera

---

## 🎨 UI Color Palette

The interface uses an Apple-inspired color scheme:

| Color | Hex | Usage |
|-------|-----|-------|
| iOS Blue | `#007AFF` | Primary accent, focus ring |
| iOS Green | `#34C759` | High focus state |
| iOS Orange | `#FF9500` | Low focus state |
| iOS Red | `#FF3B30` | No focus state |
| iOS Purple | `#AF52DE` | Eye metrics |
| iOS Cyan | `#5AC8FA` | Gaze metrics |

---

## 📋 Grading Scale

| Grade | Score Range | Description |
|-------|-------------|-------------|
| A+ | 90-100 | Exceptional focus |
| A | 80-89 | Excellent focus |
| B | 70-79 | Good focus |
| C | 60-69 | Average focus |
| D | 50-59 | Below average |
| F | 0-49 | Poor focus |

---

## 🔮 Future Improvements

- [ ] Custom calibration profiles
- [ ] Multiple session history tracking
- [ ] Focus patterns and trends analysis
- [ ] Pomodoro timer integration
- [ ] Sound/notification alerts
- [ ] Export to CSV/JSON formats
- [ ] Multi-monitor support

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **MediaPipe** by Google for face mesh detection
- **OpenCV** for computer vision operations
- **Matplotlib** for report generation

---

<p align="center">
  <b>Built with ❤️ for productivity enthusiasts</b>
</p>

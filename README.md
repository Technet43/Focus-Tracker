# 🎯 AI Focus Tracker

> Real-time attention monitoring system using computer vision and deep learning

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-ML-00A67E?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)

<p align="center">
  <img src="demo_screenshot.png" alt="Focus Tracker Demo" width="800">
</p>


## 📌 Project Overview

A sophisticated **real-time focus tracking application** that leverages computer vision and machine learning to analyze user attention levels during work or study sessions. The system processes live webcam feed to detect gaze direction, head pose, and eye states, providing instant feedback through a modern glassmorphism UI.

**Key Achievement:** Achieved accurate focus state classification by implementing a weighted multi-metric scoring algorithm with hysteresis-based state smoothing.

---

## 🛠 Technical Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Computer Vision** | OpenCV, MediaPipe Face Mesh |
| **ML/AI** | MediaPipe (468 3D facial landmarks) |
| **Data Processing** | NumPy, Collections (deque) |
| **Visualization** | Matplotlib, Custom OpenCV rendering |
| **GUI Framework** | Tkinter |
| **Report Generation** | Matplotlib PDF/PNG export |

---

## ⚡ Key Features & Technical Implementation

### 1. Multi-Modal Focus Detection

```
Focus Score = (Gaze × 0.48) + (Eye Openness × 0.47) + (Head Pose × 0.05)
```

- **Iris Tracking:** Calculates normalized iris position within eye boundaries using MediaPipe's iris landmarks (468-point face mesh)
- **Head Pose Estimation:** Implements PnP (Perspective-n-Point) algorithm with `cv2.solvePnP()` to extract Euler angles (yaw, pitch, roll)
- **Eye Aspect Ratio (EAR):** Monitors blink patterns and drowsiness using geometric eye landmark ratios

### 2. Intelligent State Management

- **Hysteresis Thresholds:** Prevents state flickering with dual-threshold system (HIGH_ON=80, HIGH_OFF=72)
- **Temporal Smoothing:** 15-frame sliding window with 75% consensus requirement for state transitions
- **Liveness Detection:** Identifies static images/photos by monitoring iris movement variance over time

### 3. Real-Time Performance Optimization

- Efficient frame processing pipeline maintaining 30+ FPS
- Deque-based rolling buffers for O(1) metric updates
- Selective UI rendering to minimize computational overhead

### 4. Computer Vision Pipeline

```
Camera Feed → MediaPipe Face Mesh → Landmark Extraction → 
  ├── Gaze Analysis (Iris position normalization)
  ├── Head Pose (3D-2D PnP solving)
  ├── Eye State (EAR calculation)
  └── Weighted Score → State Classification → UI Rendering
```

---

## 🏗 Architecture

```
Focus_Tracker/
├── Focus_Tracker_Beta.py      # Main application (1400+ lines)
│   ├── ModernStartDialog      # Tkinter-based launch interface
│   ├── UI Drawing Functions   # Glassmorphism rendering system
│   │   ├── draw_rounded_rect()
│   │   ├── draw_glass_panel()
│   │   ├── draw_progress_ring()
│   │   └── draw_mini_chart()
│   ├── Core Analysis          # ML-powered focus detection
│   │   ├── calculate_gaze_score()
│   │   ├── calculate_head_score()
│   │   ├── get_head_pose_angles()
│   │   └── analyze_focus()
│   └── Report Generator       # Matplotlib-based PDF/PNG export
└── README.md
```

---

## 📊 Algorithms & Methods

### Gaze Score Calculation
Normalizes iris center position relative to eye boundaries, computing radial distance from center:

```python
dx = abs(normalized_x - 0.5) / 0.5
dy = abs(normalized_y - 0.5) / 0.5
r = sqrt(dx² + dy²)
score = 100 × (1 - (r - r0) / (r1 - r0))  # Linear falloff
```

### Head Pose Estimation
Uses 6-point facial model with PnP algorithm:
- Model points: Nose tip, chin, eye corners, mouth corners
- Camera matrix: Constructed from frame dimensions
- Output: Euler angles via `cv2.decomposeProjectionMatrix()`

### Final Score Formula
```python
final = (avg_score × 0.85) + (high_focus_pct × 0.15) - normalized_penalties
```

---

## 🎨 UI/UX Design

Implemented a modern **Apple-inspired glassmorphism** interface:

- **Dark theme** with semi-transparent panels (alpha blending)
- **Real-time metrics** with animated progress rings
- **Live sparkline chart** showing 30-point score history
- **Color-coded status pills** for instant state recognition
- **Responsive scaling** based on window dimensions

| State | Color | Threshold |
|-------|-------|-----------|
| High Focus | `#34C759` (Green) | ≥80% |
| Low Focus | `#FF9500` (Orange) | 30-79% |
| No Focus | `#FF3B30` (Red) | <30% |

---

## 📈 Output & Analytics

The system generates comprehensive PDF/PNG reports including:

- **Focus Score Timeline** — Temporal visualization of attention levels
- **State Distribution** — Pie/bar chart of time allocation
- **Session Statistics** — Duration, averages, event counts
- **Performance Grade** — A+ to F rating system

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install opencv-python mediapipe numpy matplotlib

# Run application
python Focus_Tracker_Beta.py
```

**Controls:**
- `Enter` — Start session
- `Q` — End session & generate report
- `Esc` — Cancel

---

## 📋 Results & Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Reliable state detection with <2s latency |
| **Frame Rate** | Consistent 30+ FPS on standard hardware |
| **Detection Range** | 30-80cm optimal distance |
| **Light Tolerance** | Functions in varied lighting conditions |

---

## 🔮 Future Roadmap

- [ ] LSTM-based temporal pattern recognition
- [ ] Multi-face tracking support
- [ ] Cloud sync for cross-device analytics
- [ ] Browser extension integration
- [ ] Mobile (Android) port with CameraX

---

## 🎓 Learning Outcomes

Through this project, I developed expertise in:

- **Computer Vision:** Real-time video processing, landmark detection, pose estimation
- **Machine Learning Integration:** Implementing pre-trained models (MediaPipe) in production pipelines
- **Algorithm Design:** Multi-metric scoring systems, hysteresis-based state machines
- **Software Engineering:** Modular architecture, performance optimization, cross-platform compatibility
- **UI/UX Development:** Custom rendering systems, glassmorphism design patterns

---

## 👤 Author

**Burak**  
Electrical-Electronics Engineering Student  
Marmara University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/yourprofile)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Built as a personal productivity tool and portfolio project demonstrating computer vision expertise.</i>
</p>

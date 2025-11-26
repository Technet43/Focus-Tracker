# 🎯 Ultra AI Focus Tracker v4.0

AI-powered focus tracking desktop app with **gaze detection**, **head pose estimation**, **eye-openness / yawn analysis**, and an **Apple-inspired modern UI**.  
At the end of every session, it automatically generates a clean **PDF + PNG productivity report** in your Downloads folder.

---

## ✨ Overview

Ultra AI Focus Tracker turns your webcam into a **real-time focus monitor**.

During a session it analyses:

- Where you look (gaze & iris position)
- How open your eyes are (blink / drowsiness)
- Head orientation (yaw–pitch–roll)
- Yawns
- Distraction events and phone-check patterns
- Overall focus state over time

At the end it summarizes everything as:

- A **final focus score** (0–100)
- A **letter grade** (A+ / A / B / …)
- A detailed visual **timeline report**

The project was built as a **personal productivity tool** and as a **portfolio project** for internships.

---

## 🧠 Key Features

### 1. Real-time Focus Analysis

- Face tracking with **MediaPipe Face Mesh**
- Gaze scoring based on iris position inside the eye region
- Eye Aspect Ratio (EAR) for blink / eye-closure detection
- Mouth Aspect Ratio (MAR) for yawn detection
- Head pose estimation (3D model → yaw, pitch, roll)
- Liveness check (detects “no movement” / static face)

### 2. Intelligent Focus Scoring

The app combines three main metrics:

- Gaze score  
- Head pose score  
- Eye-openness score  

into a single focus value:

\[
\text{focus\_score} = 0.48 \cdot \text{gaze} + 0.05 \cdot \text{head} + 0.47 \cdot \text{eyes}
\]

Then it:

- Smooths state transitions using a **state buffer**
- Classifies each moment as:
  - `high_focus`
  - `low_focus`
  - `no_focus / away`
- Counts **distraction events** and **phone-checks** based on pose and gaze
- Applies penalties for yawns and distractions when computing the final session score

### 3. Apple-Inspired Modern UI

Custom-drawn UI using OpenCV:

- Glassmorphism-style panels
- Rounded cards and focus “pills”
- Live mini chart (sparkline) of the focus score
- Per-metric progress bars (Gaze / Head Pose / Eyes)
- Session stats panel showing:
  - High / Low / Away times (seconds)
  - Distractions
  - Yawns
- Center-bottom hint overlay: `Press Q to quit and save report`

### 4. Automatic Session Report

When you press **Q** to end a session, the app creates a report in your **Downloads** folder:

- `FocusTracker_Report_YYYY-MM-DD_HH-MM-SS.png`
- `FocusTracker_Report_YYYY-MM-DD_HH-MM-SS.pdf`

The report includes:

- Final focus score + letter grade
- Session duration
- Average focus score
- Percentage of time in High / Low / Away
- Distraction and yawn counts
- Focus score over time (line chart)
- Gaze vs head-pose sub-metric charts
- Color-coded focus-state timeline

---

## 🛠 Tech Stack

- **Language:** Python  
- **Computer Vision:** OpenCV, MediaPipe Face Mesh  
- **Math / Data:** NumPy  
- **Visualization & Reports:** Matplotlib, PdfPages  
- **Desktop UI:** Tkinter (modern start dialog)  
- **OS Integration:** Downloads folder detection for Windows / macOS / Linux  

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/Focus-Tracker.git
cd Focus-Tracker
(Optional) Create and activate a virtual environment.

Install dependencies:

bash
Copy code
pip install opencv-python mediapipe matplotlib numpy
Tkinter is usually included with standard Python distributions.
If it is missing, install it via your OS package manager (for example on Ubuntu):

bash
Copy code
sudo apt install python3-tk
▶️ Usage
Run the main script:

bash
Copy code
python Focus_Tracker_Beta.py
Typical workflow:

A modern start dialog opens.

Click “Start Session” (or press Enter).

The camera window appears and shows:

Live focus score

Gaze / Head / Eyes metrics

Mini focus chart

Session stats (High / Low / Away / Distractions / Yawns)

When you are done, press Q.

A PNG + PDF report is generated in your Downloads folder.

A popup displays your final score and grade.

🧩 Project Structure
text
Copy code
Focus_Tracker_Beta.py   # Main script: UI, tracking, scoring, reporting
In the future this can be split into modules (e.g. ui.py, metrics.py, report.py) if the project grows.

🚀 Possible Future Improvements
Export raw time-series data as CSV

Multi-session dashboard to compare days / weeks

User-customizable thresholds and sensitivity

Light/Dark theme toggle in the live UI

Packaging as a standalone desktop app (PyInstaller)

📄 License
This project is licensed under the MIT License.
See the LICENSE file for more details.

👤 Author
Ömer Burak

GitHub: @<your-username>

LinkedIn: <your LinkedIn profile URL>

If you find this project useful, consider giving it a ⭐ on GitHub!

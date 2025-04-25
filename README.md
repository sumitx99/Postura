# 🏋️‍♀️ PosturePerfect - AI-Based Exercise Form Tracker

> An intelligent and interactive system that detects your exercises, monitors your posture, and provides real-time feedback — turning your webcam into a personal fitness coach.

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/03a46e25-8d5a-42d6-8b7b-d3f6c58650af" alt="giphy" width="500">
</p>

## 🚀 Features


### 🔍 Auto-Detection of Exercises
- Automatically detects which of the **10 supported exercises** you're performing
- Uses real-time **pose estimation and joint angle pattern recognition**

### 🏋️‍♂️ Supported Exercises
- Push-ups, Squats, Plank, Lunges  
- Bicep Curls, Shoulder Press, Jumping Jacks  
- Sit-ups, Lateral Raises, Tricep Extensions

### 🧠 Smart Feedback System
- Real-time **form feedback** with color-coded posture warnings
- Visual display of **target muscle groups**
- **Tutorial Mode** to teach proper form before starting
- On-screen skeleton and joint angle overlays

### 📊 Advanced Metrics Tracking
- **Form Quality Score**: Drops with poor posture
- **Rep Counter** with timing analysis
- **Calories Burned** estimation based on exercise & reps
- **Session Summary** and workout history

### 📁 Data Logging
- All session data is saved in **CSV format** for future tracking

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the app |
| `R` | Reset counters |
| `A` | Toggle auto-detection |
| `T` | Toggle tutorial mode |
| `S` | Show/hide skeleton |
| `D` | Show/hide angle measurements |
| `1-0` | Manually select an exercise |

---

## 📸 How to Use

1. Run the program.
2. Choose an exercise or enable auto-detection.
3. Stand in front of your webcam.
4. Begin exercising — the system will:
   - Count your reps
   - Score your form
   - Provide visual & verbal cues
5. Use keyboard shortcuts to enhance your session.

<p align="center">
  🎥 <a href="https://github.com/user-attachments/assets/31b1ba04-d332-4c43-af40-fd169d16446f">Click here to watch the demo video</a>
</p>

---
## 🧠 Tech Stack

- **OpenCV** – for camera input and visual overlays  
- **MediaPipe** – for pose estimation and joint detection  
- **NumPy** & **Pandas** – for data analysis and logging  
- **Tkinter** / **PyQt** *(if GUI-based)* – for interactive UI  
- **Matplotlib** *(optional)* – for visualizing performance stats

---

## 📈 Future Improvements

- 🎤 Voice feedback for posture correction  
- 💤 Fatigue detection and rest suggestion  
- 🏗️ Custom workout builder (circuit-style)  
- 🧑‍🤝‍🧑 Multiplayer / partner workout sync  
- 🏆 Gamified XP system, badges, and streak tracking

<p align="center">
  <img src="https://github.com/user-attachments/assets/201ea99c-7ca8-4ffc-bc83-ff55750bb8cf" alt="giphy" width="500">
</p>

---


## 📜 License

This project is licensed under the **MIT License**.  
© 2025 [Sumit Kumar Ranjan]

---

## 🙌 Acknowledgments

Built with a passion for fitness and a fascination with computer vision 💪📷  
Big thanks to the open-source community and the creators of MediaPipe and OpenCV.

# 🌐 Quantum Invisibility Portal

An interactive, AI-powered real-time video manipulation web application built with Streamlit and MediaPipe. By detecting hand gestures via a live webcam feed, the application creates a dynamic "invisibility window" between your hands, masking out your body and revealing the calibrated background behind you.


![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00C7B7?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![WebRTC](https://img.shields.io/badge/WebRTC-333333?style=for-the-badge&logo=webrtc&logoColor=white)

---

## ✨ Features

* **Pinch-to-Activate Portal:** Pinch the thumb and index finger of **BOTH** hands simultaneously to dynamically spawn and size the rectangular invisibility zone.
* **Intelligent Selfie Segmentation:** Uses MediaPipe's computer vision mapping to target and erase the user inside the portal box while keeping static objects intact.
* **WebRTC Integration:** Powered by `streamlit-webrtc` alongside custom fallback STUN/TURN configurations for seamless video streaming over secure connections and firewalls.
* **Thread-Safe Calibration:** Built using memory-isolated state tracking to ensure zero-lag frame rendering and real-time responsiveness.
* **Responsive Layout:** Features a mobile-optimized resolution layout and a dedicated full-width background recalibration suite.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend UI:** [Streamlit](https://streamlit.io/)
* **Video Streaming:** [Streamlit-WebRTC](https://github.com/whitphx/streamlit-webrtc) & PyAV
* **AI Models:** [Google MediaPipe](https://github.com/google-ai-edge/mediapipe) (Hands & Selfie Segmentation API)
* **Image Processing:** OpenCV (`opencv-python`) & NumPy

---

## 🚀 Quick Start (Local Setup)

Want to run this project on your local machine? Follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/techiee-jhalak/Invisible-Ghost-Project.git](https://github.com/techiee-jhalak/Invisible-Ghost-Project.git)
cd Invisible-Ghost-Project

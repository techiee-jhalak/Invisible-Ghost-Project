import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

st.set_page_config(page_title="Invisibility Box Portal", layout="centered")
st.title("🌐 Quantum Invisibility Portal")
st.markdown("Pinch **BOTH** hands together (thumb & index finger) to create a dynamic invisibility window!")

# Safe cloud module mapping for MediaPipe architecture
@st.cache_resource
def load_mediapipe_models():
    # Direct access safely targets the framework components without top-level alias collision
    hands = mp.solutions.hands.Hands(
        max_num_hands=2, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
    return hands, selfie

hands, selfie = load_mediapipe_models()

def is_pinching(hand_landmarks):
    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
    distance = np.linalg.norm(thumb_tip - index_tip)
    return distance < 0.05

def draw_tech_corners(img, pt1, pt2, color, thickness, length=12):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y2 if y1 > y2 else y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

# Maintain state using Streamlit session_state memory spaces
if "background" not in st.session_state:
    st.session_state.background = None
if "calibrate_frames" not in st.session_state:
    st.session_state.calibrate_frames = 0

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    # Background Calibration
    if st.session_state.calibrate_frames < 30:
        st.session_state.background = img.copy()
        st.session_state.calibrate_frames += 1
        cv2.putText(img, f"CALIBRATING BACKGROUND... {int((st.session_state.calibrate_frames/30)*100)}%", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    hand_results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True
    
    output_frame = img.copy()
    pinch_positions = []

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            index_pos = (int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h))
            if is_pinching(hand_landmarks):
                pinch_positions.append(index_pos)
                cv2.circle(output_frame, index_pos, 5, (0, 255, 255), -1)
            else:
                cv2.circle(output_frame, index_pos, 3, (150, 150, 150), -1)

    if len(pinch_positions) == 2:
        pt1, pt2 = pinch_positions[0], pinch_positions[1]
        x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
        x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if (x2 - x1) > 15 and (y2 - y1) > 15:
            rgb_frame.flags.writeable = False
            segment_results = selfie.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            segmentation_mask = segment_results.segmentation_mask
            condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.4
            
            bg_img = st.session_state.background if st.session_state.background is not None else img
            full_invisible = np.where(condition, bg_img, img)

            box_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
            box_mask_3ch = np.stack((box_mask,) * 3, axis=-1)

            output_frame = np.where(box_mask_3ch == 255, full_invisible, img)
            draw_tech_corners(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    else:
        cv2.putText(output_frame, "PORTAL HUD: READY", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(output_frame, format="bgr24")

# Modern, non-deprecated Streamlit WebRTC configuration hook
webrtc_streamer(
    key="invisibility-portal",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.button("Reset / Recalibrate Background"):
    st.session_state.calibrate_frames = 0
    st.rerun()
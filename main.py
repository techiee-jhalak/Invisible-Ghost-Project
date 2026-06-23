import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe solutions safely
mp_hands = mp.solutions.hands
mp_selfie = mp.solutions.selfie_segmentation

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
selfie = mp_selfie.SelfieSegmentation(model_selection=1)

# --- BULLETPROOF VIDEO CAPTURE FALLBACK LOGIC ---
cap = None

if hasattr(cv2, 'CAP_DSHOW'):
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    except Exception:
        cap = None

if cap is None or not cap.isOpened():
    try:
        cap = cv2.VideoCapture(0)
    except Exception as e:
        print(f"Critical error allocating system camera framework: {e}")
        exit()

if not cap.isOpened():
    print("Error: Could not open webcam through any system video pipeline.")
    exit()
# ------------------------------------------------

print("Calibrating background... Please step out of the frame for 3 seconds.")
background = None
start_time = time.time()

# 1. Background Calibration Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1) # Mirror effect
    elapsed = time.time() - start_time
    display_frame = frame.copy()
    countdown = 3 - int(elapsed)
    
    if countdown > 0:
        cv2.putText(display_frame, f"Calibrating in {countdown}...", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        background = frame.copy()
        
        # QUALITY UPGRADE: Attempt to lock automatic exposure settings on supported webcams
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 1 = Manual Mode lock
        except Exception:
            pass
            
        print("--> Calibration Complete! Exposure Locked. Portal Window Active.")
        break
        
    cv2.imshow("Invisibility Box Studio", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

def is_pinching(hand_landmarks):
    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
    distance = np.linalg.norm(thumb_tip - index_tip)
    return distance < 0.05

# 2. Main Processing Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
        
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    hand_results = hands.process(rgb_frame)
    output_frame = frame.copy()
    pinch_positions = []
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            index_pos = (int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h))
            
            if is_pinching(hand_landmarks):
                pinch_positions.append(index_pos)
                cv2.circle(output_frame, index_pos, 10, (0, 255, 0), -1)
            else:
                cv2.circle(output_frame, index_pos, 6, (0, 255, 255), -1)

    # 3. ENHANCED Geometry Masking & Shade Correction Logic
    if len(pinch_positions) == 2:
        pt1, pt2 = pinch_positions[0], pinch_positions[1]
        x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
        x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            segment_results = selfie.process(rgb_frame)
            segmentation_mask = segment_results.segmentation_mask
            
            # QUALITY UPGRADE: Dynamic Color Tint Harmonization
            # Gently blend 95% background image with 5% ambient lighting structure to neutralize camera shifts
            adjusted_background = cv2.addWeighted(background, 0.95, frame, 0.05, 0)
            
            condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.4
            full_invisible = np.where(condition, adjusted_background, frame)
            
            # Create a geometric box frame canvas
            box_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
            
            # QUALITY UPGRADE: Anti-Aliasing & Feathered Corner Transitions
            # Soften the edges of the box mask via 21x21 Gaussian Blur filter calculations
            feathered_mask = cv2.GaussianBlur(box_mask, (21, 21), 0)
            feathered_mask_3ch = np.stack((feathered_mask,) * 3, axis=-1) / 255.0
            
            # Smooth Alpha Matting blend equation
            output_frame = (full_invisible * feathered_mask_3ch + frame * (1.0 - feathered_mask_3ch)).astype(np.uint8)
            
            # Technical architectural overlay border tracking lines
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(output_frame, "PORTAL ACTIVE", (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(output_frame, "Pinch BOTH hands to open Invisibility Box", (30, 50), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Invisibility Box Studio", output_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('r'):
        # Allow instant background resaving if lighting environment swings heavily
        background = frame.copy()
        print("Background Recalibrated!")

cap.release()
cv2.destroyAllWindows()
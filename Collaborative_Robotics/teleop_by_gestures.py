import dobotArm
import lib.DobotDllType as dType
import numpy as np
import cv2
import time
import os
import sys

import libteam21
import mediapipe as mp

try:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_drawing
except ImportError:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing

print("=== STARTING DIRECT HAND GESTURE TELEOPERATION ===")

# --- PHYSICAL WORKSPACE SAFETY LIMITS ---
X_MIN, X_MAX = 180, 280
Y_MIN, Y_MAX = -150, 150
Z_MIN, Z_MAX = -50, 70

# Default control height
current_z = 30

# --- INITIALIZATION ---
print("1. Loading Dobot API...")
api = dType.load()

print("2. Opening Camera...")
cam_index, cam_backend = libteam21.auto_select_camera("usb")
cap = cv2.VideoCapture(cam_index, cam_backend)

ret, frame = cap.read()
if not ret or frame is None:
    print(f"[WARNING] Auto-selected camera index {cam_index} failed. Falling back to index 2...")
    cap.release()
    cap = cv2.VideoCapture(2)
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[WARNING] Index 2 failed. Falling back to index 0...")
        cap.release()
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

if frame is None:
    print("Error: Could not open any camera device!")
    sys.exit(1)

# Camera transformation parameters safely loaded from either ./ or ../
if os.path.exists("HomographyMatrix.npy"):
    H_matrix = np.load("HomographyMatrix.npy")
elif os.path.exists("../HomographyMatrix.npy"):
    H_matrix = np.load("../HomographyMatrix.npy")
else:
    print("Error: HomographyMatrix.npy not found!")
    sys.exit(1)

if os.path.exists("camera_params.npz"):
    data = np.load("camera_params.npz")
elif os.path.exists("../camera_params.npz"):
    data = np.load("../camera_params.npz")
else:
    print("Error: camera_params.npz not found!")
    sys.exit(1)

camera_matrix = data["camera_matrix"]
dist_coeffs   = data["dist_coeffs"]

h, w = frame.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w,h), cv2.CV_16SC2)

# --- ROBOT INITIALIZATION ---
print("3. Initializing Robot and Homing...")
dobotArm.initialize_robot(api)
dobotArm.open_gripper(api)
dobotArm.stop_pump(api)
print("[SUCCESS] Robot is ready.")


def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]


def detect_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers_up = 0
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for tip, pip in zip(finger_tips, finger_pips):
        if lm[tip].y < lm[pip].y:
            fingers_up += 1

    thumb_index_dist = ((lm[4].x - lm[8].x) ** 2 + (lm[4].y - lm[8].y) ** 2) ** 0.5

    if fingers_up >= 4:
        return "open_palm"
    elif fingers_up == 0:
        return "fist"
    elif thumb_index_dist < 0.055:
        return "pinch"
    else:
        return "neutral"


# --- MAIN TELEOPERATION LOOP ---
print("\n=== TELEOPERATION CONTROL INTERFACE ACTIVE ===")
print("Use your hand gestures in front of the camera:")
print("  - PINCH (Thumb + Index touching): DRAG & DRIVE the robot arm gripper in real-time!")
print("  - SHOW FIST: Instantly closes the gripper (grabs target).")
print("  - SHOW OPEN PALM: Instantly opens the gripper (releases target).")
print("  - MOVE HAND HIGHER/LOWER (while pinching): Dynamically adjust effector Z height!")
print("Press Ctrl+C in the terminal to exit.")

# Exponential smoothing history
smooth_rx, smooth_ry = 200.0, 0.0
alpha = 0.35  # Smoothing factor (higher = more responsive, lower = smoother)

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            display_frame = frame.copy()
            
            # Read hands landmarks
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            status_text = "No hand detected"
            status_color = (0, 0, 255)
            
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                
                # Draw hand skeletal landmarks on screen
                mp_drawing.draw_landmarks(
                    display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                
                gesture = detect_gesture(hand_landmarks)
                
                # Get the pixel coordinates of the index finger tip (landmark 8)
                index_tip = hand_landmarks.landmark[8]
                pixel_x = int(index_tip.x * w)
                pixel_y = int(index_tip.y * h)
                
                # Draw a target circle on the index finger
                cv2.circle(display_frame, (pixel_x, pixel_y), 10, (255, 255, 0), -1)
                
                # Compute physical hand distance (depth) from table using Wrist (0) and Middle Knuckle (9) pixel span
                wrist = hand_landmarks.landmark[0]
                knuckle = hand_landmarks.landmark[9]
                wrist_px = (wrist.x * w, wrist.y * h)
                knuckle_px = (knuckle.x * w, knuckle.y * h)
                span = np.sqrt((wrist_px[0] - knuckle_px[0])**2 + (wrist_px[1] - knuckle_px[1])**2)

                if gesture == "open_palm":
                    status_text = "OPEN PALM -> Gripper OPENED"
                    status_color = (0, 255, 0)
                    dobotArm.open_gripper(api)
                    dobotArm.stop_pump(api)
                    
                elif gesture == "fist":
                    status_text = "FIST -> Gripper CLOSED"
                    status_color = (0, 0, 255)
                    dobotArm.close_gripper(api)
                else:
                    status_text = "TRACKING ACTIVE"
                    status_color = (255, 255, 0)

                # Continuous 3D hand tracking (regardless of gesture!)
                # Convert index finger tip pixel to robot world coordinate
                target_rx, target_ry = pixel_to_robot(pixel_x, pixel_y, H_matrix)
                
                # Map the hand's pixel span (proximity to camera) to physical Z distance from the table!
                SPAN_MIN, SPAN_MAX = 50.0, 130.0
                clamped_span = max(SPAN_MIN, min(SPAN_MAX, span))
                target_rz = Z_MIN + ((clamped_span - SPAN_MIN) / (SPAN_MAX - SPAN_MIN)) * (Z_MAX - Z_MIN)
                
                # Apply safety workspace bounding box limits
                target_rx = max(X_MIN, min(X_MAX, target_rx))
                target_ry = max(Y_MIN, min(Y_MAX, target_ry))
                target_rz = max(Z_MIN, min(Z_MAX, target_rz))
                
                # Apply exponential moving average smoothing to prevent jumping
                smooth_rx = (1 - alpha) * smooth_rx + alpha * target_rx
                smooth_ry = (1 - alpha) * smooth_ry + alpha * target_ry
                current_z = (1 - alpha) * current_z + alpha * target_rz
                
                # Issue immediate, non-blocking move command
                # isQueued=0 means do not append to queue, execute instantly!
                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, smooth_rx, smooth_ry, current_z, 0, isQueued=0)
                
                # Draw movement vector lines on GUI
                cv2.putText(display_frame, f"Robot Pos: ({smooth_rx:.0f}, {smooth_ry:.0f}, {current_z:.0f})", 
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                status_text = f"Neutral Gesture (No Hand)"
                status_color = (128, 255, 128)
            
            # Overlay HUD Info on GUI
            cv2.putText(display_frame, f"GESTURE: {status_text}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_frame, "STANDALONE TELEOPERATION CONTROLLER", (20, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show live window
            cv2.imshow("Direct Gesture Teleoperation", display_frame)
            cv2.waitKey(1)

except KeyboardInterrupt:
    print("\nTeleoperation stopped by user.")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

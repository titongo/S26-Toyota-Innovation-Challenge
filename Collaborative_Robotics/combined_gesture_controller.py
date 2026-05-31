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

print("=== STARTING UNIFIED COBOT GESTURE SYSTEM ===")

# --- PHYSICAL WORKSPACE SAFETY LIMITS ---
X_MIN, X_MAX = 180, 280
Y_MIN, Y_MAX = -150, 150
Z_MIN, Z_MAX = -50, 70

Z_SAFE = 40

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

# Initialize a single global MediaPipe Hands tracker to prevent massive CPU re-initialization lag
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

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

    # Prioritize pinch check first, as other fingers are curled during a pinch (which would otherwise trigger 'fist')
    if thumb_index_dist < 0.055:
        return "pinch"
    elif fingers_up >= 4:
        return "open_palm"
    elif fingers_up == 0:
        return "fist"
    else:
        return "neutral"


# --- MOTION STATE VARIABLES ---
current_mode = "Patrolling"  # Options: "Patrolling", "Paused", "Teleoperating"
smooth_rx, smooth_ry = 200.0, 0.0
current_z = 30
alpha = 0.35  # Exponential smoothing factor

# State-change and throttling variables to prevent flooding the serial bus
last_gripper_state = None  # Tracks open/closed to avoid spamming gripper commands
last_sent_pos = [200.0, 0.0, 30.0]
last_command_time = 0.0

print("\n=== UNIFIED COBOT GESTURE SYSTEM ACTIVE ===")
print("Use your hand gestures in front of the camera:")
print("  - NO HAND / NEUTRAL: Robot automatically patrols back-and-forth.")
print("  - SHOW OPEN PALM: Instantly halts/pauses the robot arm movement!")
print("  - SHOW FIST: Resumes automatic patrolling (if paused).")
print("  - PINCH (Thumb + Index touching): Override & Teleoperate the gripper in real-time!")
print("Press Ctrl+C in the terminal to exit.")

# Define safe targets for patrolling
target_A = [220, 100, 30]
target_B = [220, -100, 30]
target_index = 0
targets = [target_A, target_B]

# Start active patrolling queue command
current_target = targets[target_index]
execCmd = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, current_target[0], current_target[1], current_target[2], 0, isQueued=1)[0]

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()

        # Process the hand in real-time
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        gesture = "neutral"
        status_text = "AUTO-PATROLLING..."
        status_color = (0, 255, 0)
        pixel_x, pixel_y = 0, 0
        index_tip = None

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            
            # Draw skeleton connections
            mp_drawing.draw_landmarks(
                display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            gesture = detect_gesture(hand_landmarks)

            # Get the pixel coordinates of the index finger tip (landmark 8)
            index_tip = hand_landmarks.landmark[8]
            pixel_x = int(index_tip.x * w)
            pixel_y = int(index_tip.y * h)
            cv2.circle(display_frame, (pixel_x, pixel_y), 10, (255, 255, 0), -1)

        # --- MODE TRANSITION STATE MACHINE ---
        if gesture == "pinch" and result.multi_hand_landmarks and index_tip is not None:
            # Direct Teleoperation Override Mode
            if current_mode != "Teleoperating":
                print("[OVERRIDE] Pinch detected. Switching to Interactive Teleoperation!")
                dType.SetQueuedCmdForceStopExec(api)
                current_mode = "Teleoperating"
            
            status_text = "TELEOPERATING (PINCH OVERRIDE)"
            status_color = (255, 255, 0)

            # Translate index tip pixel to robot world coordinates
            target_rx, target_ry = pixel_to_robot(pixel_x, pixel_y, H_matrix)
            target_rz = Z_MIN + (1.0 - index_tip.y) * (Z_MAX - Z_MIN)

            # Enforce safety boundaries
            target_rx = max(X_MIN, min(X_MAX, target_rx))
            target_ry = max(Y_MIN, min(Y_MAX, target_ry))
            target_rz = max(Z_MIN, min(Z_MAX, target_rz))

            # Apply smoothing
            smooth_rx = (1 - alpha) * smooth_rx + alpha * target_rx
            smooth_ry = (1 - alpha) * smooth_ry + alpha * target_ry
            current_z = (1 - alpha) * current_z + alpha * target_rz

            # Throttled non-blocking serial communication to avoid clogging the buffer
            now = time.time()
            if now - last_command_time > 0.08:  # Maximum 12 commands per second
                dist_moved = np.sqrt((smooth_rx - last_sent_pos[0])**2 + 
                                     (smooth_ry - last_sent_pos[1])**2 + 
                                     (current_z - last_sent_pos[2])**2)
                if dist_moved > 3.0:  # Only issue command if target shifted by > 3mm
                    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, smooth_rx, smooth_ry, current_z, 0, isQueued=0)
                    last_sent_pos = [smooth_rx, smooth_ry, current_z]
                    last_command_time = now

        elif gesture == "open_palm":
            # Safety Pause Mode
            if current_mode != "Paused":
                print("[SAFETY] Open palm detected. Pausing robot arm immediately!")
                dType.SetQueuedCmdForceStopExec(api)
                # Move to safe vertical height
                dobotArm.move_to_xyz(api, smooth_rx, smooth_ry, Z_SAFE)
                current_mode = "Paused"
                
            status_text = "SAFETY PAUSE (OPEN PALM)"
            status_color = (0, 0, 255)
            
            # State change lock to avoid spamming the gripper on every frame
            if last_gripper_state != "open":
                dobotArm.open_gripper(api)
                dobotArm.stop_pump(api)
                last_gripper_state = "open"

        elif gesture == "fist":
            # State change lock to avoid spamming the gripper on every frame
            if last_gripper_state != "closed":
                dobotArm.close_gripper(api)
                last_gripper_state = "closed"
                
            if current_mode == "Paused":
                print("[SAFETY] Fist gesture. Resuming patrolling...")
                current_mode = "Patrolling"
                dType.SetQueuedCmdStartExec(api)
                # Re-issue active target move
                execCmd = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, current_target[0], current_target[1], current_target[2], 0, isQueued=1)[0]
            
            status_text = "GRIPPER CLOSED"
            status_color = (128, 0, 128)

        else:
            # Neutral / No hand
            if current_mode == "Teleoperating":
                print("[OVERRIDE] Pinch released. Resuming Patrolling mode.")
                current_mode = "Patrolling"
                dType.SetQueuedCmdStartExec(api)
                # Re-issue active target move from current location
                execCmd = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, current_target[0], current_target[1], current_target[2], 0, isQueued=1)[0]
            
            if current_mode == "Patrolling":
                status_text = f"PATROLLING -> TARGET {chr(65 + target_index)}"
                status_color = (0, 255, 0)

                # Check if the active patrolling command has completed
                if execCmd <= dType.GetQueuedCmdCurrentIndex(api)[0]:
                    # Switch target A <-> B
                    target_index = 1 - target_index
                    current_target = targets[target_index]
                    print(f"[PATROL] Arrived at target. Moving to Target {chr(65 + target_index)}...")
                    execCmd = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, current_target[0], current_target[1], current_target[2], 0, isQueued=1)[0]

            elif current_mode == "Paused":
                status_text = "PAUSED - SHOW FIST TO RESUME"
                status_color = (0, 0, 255)

        # Overlay HUD on GUI
        cv2.putText(display_frame, f"MODE: {status_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(display_frame, f"Active Gesture: {gesture}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, "UNIFIED COBOT GESTURE SYSTEM", (20, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Unified Cobot Controller", display_frame)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("\nCobot controller stopped by user.")
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

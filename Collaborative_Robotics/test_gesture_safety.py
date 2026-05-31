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

# Initialize a single global MediaPipe Hands tracker to prevent massive CPU re-initialization lag
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

print("=== STARTING STANDALONE GESTURE SAFETY TEST ===")

# --- CONSTANTS ---
Z_SAFE = 40
home_pos = [200, 100, 50]

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
print("[SUCCESS] Homing complete. Robot is ready.")


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
    elif thumb_index_dist > 0.12:
        return "pinch_open"
    elif thumb_index_dist < 0.06:
        return "pinch_close"
    else:
        return "other"


def safe_move_to_xyz(api, x, y, z):
    print(f"Moving to: ({x}, {y}, {z}) with real-time gesture tracking...")
    
    # Start the immediate movement command (enqueued with isQueued=1 for reliable tracking!)
    execCmd = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, 0, isQueued=1)[0]
    
    # Continuous camera read and hand tracking loop while the robot is moving
    while execCmd > dType.GetQueuedCmdCurrentIndex(api)[0]:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()

        # Re-use the pre-initialized global hands object for blazing speed!
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            # Draw the full hand skeletal skeleton live on the GUI!
            mp_drawing.draw_landmarks(
                display_frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
            )
            
            gesture = detect_gesture(result.multi_hand_landmarks[0])

            if gesture == "pinch_open":
                print("[GESTURE] Thumb-index open detected. Opening gripper.")
                dobotArm.open_gripper(api)
            elif gesture == "pinch_close":
                print("[GESTURE] Thumb-index close detected. Closing gripper.")
                dobotArm.close_gripper(api)
            elif gesture == "open_palm":
                print("[SAFETY] Open palm detected. Pausing robot immediately!")
                
                # Force stop execution of the current movement command
                dType.SetQueuedCmdForceStopExec(api)
                
                # Raise the arm to safe height
                dobotArm.move_to_xyz(api, x, y, Z_SAFE)

                # Block in this pause loop until "fist" is shown to resume
                while True:
                    ret2, frame2 = cap.read()
                    if not ret2:
                        continue

                    frame2 = cv2.remap(frame2, map1, map2, cv2.INTER_LINEAR)
                    display_frame_paused = frame2.copy()
                    
                    result2 = hands.process(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

                    if result2.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            display_frame_paused, result2.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                        )
                        gesture2 = detect_gesture(result2.multi_hand_landmarks[0])

                        if gesture2 == "pinch_open":
                            print("[GESTURE] Thumb-index open detected. Opening gripper.")
                            dobotArm.open_gripper(api)
                        elif gesture2 == "pinch_close":
                            print("[GESTURE] Thumb-index close detected. Closing gripper.")
                            dobotArm.close_gripper(api)
                        elif gesture2 == "fist":
                            print("[SAFETY] Fist detected. Resuming movement.")
                            break
                        
                        cv2.putText(display_frame_paused, f"PAUSED - SHOW FIST TO RESUME (Current: {gesture2})", 
                                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.putText(display_frame_paused, "PAUSED - SHOW FIST TO RESUME (No Hand)", 
                                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                    cv2.imshow("Gesture Safety Window", display_frame_paused)
                    cv2.waitKey(1)

                # Resume the queue execution and restart the move command
                print("Resuming movement...")
                dType.SetQueuedCmdStartExec(api)
                execCmd = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, 0, isQueued=1)[0]

        cv2.putText(display_frame, "ROBOT MOVING...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Gesture Safety Window", display_frame)
        cv2.waitKey(1)


# --- MAIN TEST LOOP ---
print("\n=== STARTING MOTION PATTERN ===")
print("The robot will now move back and forth between two points indefinitely.")
print("Place your hand in front of the camera to test:")
print("  - SHOW OPEN PALM: Stops the robot instantly and raises the arm to safety height.")
print("  - SHOW FIST (after pause): Resumes the robot movement.")
print("  - PINCH CLOSE / PINCH OPEN: Manually controls the gripper.")
print("Press Ctrl+C in the terminal to exit.")

# Define two safe targets in workspace
target_A = [220, 120, 40]
target_B = [220, -120, 40]

try:
    while True:
        # Move to Target A
        safe_move_to_xyz(api, target_A[0], target_A[1], target_A[2])
        time.sleep(0.5)
        
        # Move to Target B
        safe_move_to_xyz(api, target_B[0], target_B[1], target_B[2])
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nTest stopped by user.")
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

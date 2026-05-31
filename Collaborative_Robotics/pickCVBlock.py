#This code is a simplified implementation of a collaborative robotics system that detects plates and targets using computer vision, 
#and then commands a Dobot robotic arm to pick and place objects accordingly. The system operates in three phases: scanning for plates, 
#scanning for targets, and executing the pick/place operations. 
#Stability checks are implemented to ensure reliable detection before proceeding to the next phase.

# Note: there are parameters that are useful to the successful operation of the robot arm. Read through the code before running the program.

# How to use: 
# 1. Ensure you have the Dobot robotic arm set up and connected to your computer.
# 2. Place the plates (drop zones) and targets (red blocks) within the camera's
# field of view.
# 3. Run the script. The system will first scan for plates, then targets, and finally execute the pick/place operations based on the detected positions.
# 4. Monitor the console output and the video feed for feedback on the system's status and operations

#Other Useful Codes you can use:
#dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE, rHead): moves the robot to the specified (x, y, z) coordinates with a specified rotation for the end effector (rHead). Z_SAFE is a predefined constant that ensures the robot maintains a safe height to avoid collisions when moving horizontally.


from Part import Part
import dobotArm
import lib.DobotDllType as dType
import numpy as np
import cv2
import time
import os
import sys

import libteam21

"""CONSTANTS"""

Z_SAFE = 40 #what is the clearance distance for the robot arm to avoid collisions when moving horizontally?
Z_PICK = -64 #what is the  height for the robot claw to successfully pick up the target?
STABILITY_LIMIT = 60  #how many consecutive frames of stable detection before we "lock in" the positions and move to the next phase? (at 30fps, 60 frames is about 2 seconds)
PIXEL_TOLERANCE = 10  #object can move at most this # of pixels to be considered stationary

#Angle to calibrate grip
GRIPPER_ANGLE_OFFSET = -45   # degrees.

machine_state = "scanning plate"

# --- INITIALIZATION FOR CAMERA TRANSFORMATION ---
# MAKE SURE THAT YOU HAVE RAN calibrateCamera.py FIRST TO GENERATE THE camera_params.npz FILE
api = dType.load()

cam_index, cam_backend = libteam21.auto_select_camera("usb")
cap = cv2.VideoCapture(cam_index, cam_backend)

# If the auto-selected camera fails to read a frame, fall back to standard indices
ret, frame = cap.read()
if not ret or frame is None:
    print(f"[WARNING] Auto-selected camera index {cam_index} failed to read. Falling back to index 2 (standard USB camera)...")
    cap.release()
    cap = cv2.VideoCapture(2) # Fall back to standard V4L2 index 2
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[WARNING] Index 2 failed. Falling back to index 0 (default webcam)...")
        cap.release()
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

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

# Compute undistort maps once
h, w = frame.shape[:2] if frame is not None else (480, 640)
new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w,h), cv2.CV_16SC2)

# Init the focus area values
x_focus, y_focus, w_focus, h_focus = [200, 100, 300, 200]  # Standard focus region

def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]

def fold_angle(a):
    # Fold into (-90, 90]. A 180-degree flip grips identically with a 2-jaw gripper,
    # and this keeps rHead within the servo's range and avoids needless big rotations.
    while a > 90:   a -= 180
    while a <= -90: a += 180
    return a


def safe_move_to_xyz(api, x, y, z, rHead=0):
    print(f"Moving to: ({x}, {y}, {z}, rot={rHead}) with real-time video refresh...")
    
    # Enqueue command cleanly using isQueued=1 to track transit progress
    execCmd = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, rHead, isQueued=1)[0]
    
    # Continuous camera read and GUI waitKey pump loop while the robot is moving
    # This prevents the operating system's window manager from flagging the window as "Not Responding" and freezing/grey-outing!
    # Contains absolutely 0% MediaPipe hand tracking or ML processing to prevent any latency, freezes, or thread deadlocks!
    while execCmd > dType.GetQueuedCmdCurrentIndex(api)[0]:
        ret, frame = cap.read()
        if not ret or frame is None:
            # If camera fails or is released, just pump Qt event loop via pollKey to keep window alive
            cv2.pollKey()
            dType.dSleep(25)
            continue

        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()

        cv2.putText(display_frame, "ROBOT MOVING...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Detection", display_frame)
        cv2.waitKey(1)


# State machine logic to control the flow of the program through the three phases: scanning for plates, scanning for targets, and executing pick/place operations.
# THIS STATE MACHINE IS TOO SIMPLE. Can you think of logics that should change the robot's sequnece of actions?
# Ex: what if the robot fails to pick up a target? should it retry? should it go back to scanning for targets in case the target was moved? what if a new plate is added during the pick/place phase?
# What if a human's hand is in sight during pick/place phase? (safety first!)

def next_state():
    global machine_state
    if machine_state == "scanning plate":
        machine_state = "scanning target"
    elif machine_state == "scanning target":
        machine_state = "pick place"
    elif machine_state == "pick place":
        machine_state = "scanning plate"
    else:
        machine_state = "scanning plate"



# ---------------------------------------------------------
# PHASE 1: DETECT Part Drop Zones (Plates)
# this script assumes a metallic circular plate as the drop zone, but you can modify the detection logic to fit your specific use case.
# ---------------------------------------------------------
def phase_detect_plates():
    print("\n[PHASE 1] Scanning for drop zones (using EDCircles). Waiting for stability...")
    stability_counter = 0
    last_count = 0
    frame_count = 0  # Counter to prevent flooding console with debug prints
    
    # Bilateral filter local tuning parameters for shiny metal disk detection
    bilateral_diameter = 12
    bilateral_sigma_color = 40
    bilateral_sigma_space = 40
    
    # Sliding window coordinate history to filter out frame-by-frame jittering
    coordinate_history = []
    
    # Initialize Edge Drawing object for EDCircles
    ed = cv2.ximgproc.createEdgeDrawing()
    
    # Universal parameters initialization (works perfectly on both Windows and Linux)
    params = ed.Params()
    params.EdgeDetectionOperator = cv2.ximgproc.EdgeDrawing_SOBEL
    params.GradientThresholdValue = 6  # Lower to capture softer/reflective metallic edges
    params.AnchorThresholdValue = 2     # Lower to extract more edge anchors
    params.NFAValidation = False        # Disable strict false alarm validation to detect shiny/broken discs
    params.MinPathLength = 10
    ed.setParams(params)
    
    while True:
        frame_count += 1
        log_debug = (frame_count % 30 == 0) # Log every 30 frames (~once per second)
        
        if log_debug:
            print("\n--- [EDCircles Debug Log] ---")
        ret, frame = cap.read()
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        frame = libteam21.boost_saturation(frame)
        display_frame = frame.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))     

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gaussian = cv2.GaussianBlur(hsv, (7, 7), 1.5)      # or cv2.medianBlur(img, 5)

        blurred = cv2.bilateralFilter(gaussian, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)
        cv2.imshow("Blurred (Debug)", blurred)
        # set region of interest
        x, y, w, h = [200, 100, 300, 200]  # adjust these values

        roi = blurred[y:y+h, x:x+w]
        
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Run EDCircles on the ROI
        ed.detectEdges(roi)
        ellipses = ed.detectEllipses()
        
        # Show debug edge map to see live edge segment extraction
        edge_img = ed.getEdgeImage()
        cv2.imshow("ED Edge Map (Debug)", edge_img)

        current_list = []
        if ellipses is not None:
            if log_debug:
                print(f"Total shapes detected by EDCircles: {len(ellipses[0])}")
                
            for idx, ellipse in enumerate(ellipses[0]):
                is_pure_circle = ellipse[2] > 0
                
                if is_pure_circle:
                    cx_roi, cy_roi, r = int(ellipse[0]), int(ellipse[1]), int(ellipse[2])
                    if log_debug:
                        print(f"  [{idx}] Pure Circle at ({cx_roi}, {cy_roi}) with radius {r}px")
                else:
                    r_major, r_minor = ellipse[3], ellipse[4]
                    ratio = r_major / r_minor if r_minor > 0 else 0
                    cx_roi, cy_roi, r = int(ellipse[0]), int(ellipse[1]), int((r_major + r_minor) / 2)
                    
                    if log_debug:
                        print(f"  [{idx}] Ellipse at ({cx_roi}, {cy_roi}): major={r_major:.1f}px, minor={r_minor:.1f}px, ratio={ratio:.2f}")
                    
                    if r_minor <= 0 or ratio >= 1.2:
                        if log_debug:
                            print(f"      -> REJECTED: Not circular enough (ratio {ratio:.2f} >= 1.2)")
                        continue
                
                # Filter circles in the expected radius range (18 to 55 pixels)
                if 18 <= r <= 55:
                    full_x = cx_roi + x      # Add ROI x offset (200)
                    
                    full_y = cy_roi + y      # Add ROI y offset (100)
                    
                    cv2.circle(display_frame, (full_x, full_y), r, (0, 255, 0), 2)
                    rx, ry = pixel_to_robot(full_x, full_y, H_matrix)
                    
                    # Apply moving average filter to smooth coordinates and prevent jitter
                    coordinate_history.append((rx, ry))
                    if len(coordinate_history) > 6:
                        coordinate_history.pop(0)  # Keep the last 6 frames of history
                    
                    smooth_rx = sum(p[0] for p in coordinate_history) / len(coordinate_history)
                    smooth_ry = sum(p[1] for p in coordinate_history) / len(coordinate_history)
                    
                    current_list.append((smooth_rx, smooth_ry))
                    if log_debug:
                        print(f"      -> ACCEPTED: Radius {r}px in range [18, 55], smoothed robot pos=({smooth_rx:.1f}, {smooth_ry:.1f})")
                else:
                    if log_debug:
                        print(f"      -> REJECTED: Radius {r}px is outside expected range [18, 55]")
        else:
            if log_debug:
                print("No shapes detected by Edge Drawing algorithm. (Tip: Try adjusting lighting, focus, or lowering GradientThresholdValue)")

        # --- AUTO-LOCK LOGIC ---
        if len(current_list) > 0 and len(current_list) == last_count:
            stability_counter += 1
        else:
            stability_counter = 0
            last_count = len(current_list)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        cv2.putText(display_frame, f"LOCKING PLATES: {progress}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Detection", display_frame)
        cv2.waitKey(1)

        if stability_counter >= STABILITY_LIMIT:
            print(f"Locked {len(current_list)} plates.")
            return current_list
  
 

# ---------------------------------------------------------
# PHASE 2: DETECT Red velcros to pick up (Red Blocks)
# ---------------------------------------------------------
def phase_detect_targets(targetPart: Part):
    print("\n[PHASE 2] Scanning for targets. Waiting for stability...")
    stability_counter = 0
    last_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        frame = libteam21.boost_saturation(frame)
        # Create a display copy so drawings don't affect next frame's HSV detection
        display_frame = frame.copy()
        
        focus_area = display_frame[y_focus:y_focus+h_focus, x_focus:x_focus+w_focus]
        targetPart.updateFrame(focus_area)
        cv2.rectangle(display_frame, (x_focus, y_focus), (x_focus+w_focus, y_focus+h_focus), (0, 255, 0), 2)
        
        # Red Tag Logic
        mask = targetPart.returnMask()
        gitmask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_list = []
        for cnt in contours:
            if cv2.contourArea(cnt) > targetPart.minContourArea:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    
                    # --- This creates a bounding box rectangle, and gives back its dimensions
                    rect = cv2.minAreaRect(cnt)          # ((px,py),(w,h),angle) in PIXELS
                    box  = cv2.boxPoints(rect)           # 4 corner points (float)
                    
                    # Calculation for finding the angle, using the shorter edge, to close upon
                    e1, e2 = box[1] - box[0], box[2] - box[1]
                    short_vec = e1 if np.linalg.norm(e1) < np.linalg.norm(e2) else e2
                    n = np.linalg.norm(short_vec)
                    if n < 1e-6:
                        continue
                    short_vec /= n      # (unit) vector
                    
                    # Measure the angle in ROBOT space, not pixels: map the centre and a
                    # point one step along the short axis through the homography, then take
                    # the angle between them in mm. This auto-corrects for camera rotation.
                    STEP = 20  # pixels
                    rx0, ry0 = pixel_to_robot(cx + x_focus, cy + y_focus, H_matrix)
                    rx1, ry1 = pixel_to_robot(cx + x_focus + short_vec[0]*STEP,
                                            cy + y_focus + short_vec[1]*STEP, H_matrix) # creates a delta-r
                    grip_angle = fold_angle(np.degrees(np.arctan2(ry1 - ry0, rx1 - rx0))
                                            + GRIPPER_ANGLE_OFFSET) #This find the angle using trig
                    
                    current_list.append((rx0, ry0, grip_angle))   # <-- now a 3-tuple
                    
                    # visual feedback
                    box_full = (box + np.array([x_focus, y_focus])).astype(np.int32)
                    cv2.drawContours(display_frame, [box_full], -1, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{grip_angle:.0f}", (cx + x_focus, cy + y_focus),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
        cv2.waitKey(1)

        # --- STABILITY LOGIC ---
        if len(current_list) != 0:
            if len(current_list) > 0 and len(current_list) == last_count:
                stability_counter += 1
            else:
                stability_counter = 0
                last_count = len(current_list)

        # Visual Feedback
        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        color = (0, 255, 0) if progress < 100 else (255, 255, 0)
        
        cv2.putText(display_frame, f"LOCKING TARGETS: {progress}%", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Detection", display_frame)
        
        # --- EXIT CONDITION ---
        if stability_counter >= STABILITY_LIMIT:
            print(f"[SUCCESS] Locked {len(current_list)} targets.")
            return current_list


# ---------------------------------------------------------
# PHASE 3: PICK/PLACE LOOP
# This function assumes 1 drop zone only has 1 part, and executes the pick/place operations in batches.
# ---------------------------------------------------------
def phase_execute_batch(api, pick_list, drop_list):
    if len(pick_list) == 0 or len(drop_list) == 0:
        print("missing targets, aborting")
        return False
    
    # Match 1 part to 1 drop zone (uses the smaller count)
    batch_size = min(len(pick_list), len(drop_list))
    print(f"\n[PHASE 3] Executing batch of {batch_size} operations.")

    for i in range(batch_size):
        pick_x, pick_y, pick_angle = pick_list[i]   # angle value added
        drop_x, drop_y = drop_list[i]

        print(f"Task {i+1}: Moving {pick_x, pick_y} to {drop_x, drop_y}, rotated to angle {pick_angle}")

        # --- PICK SEQUENCE ---
        safe_move_to_xyz(api, pick_x, pick_y, Z_SAFE, pick_angle)
        safe_move_to_xyz(api, pick_x, pick_y, Z_PICK, pick_angle)

        dobotArm.close_gripper(api)
        safe_move_to_xyz(api, pick_x, pick_y, Z_SAFE, pick_angle)

        # --- PLACE SEQUENCE ---
        safe_move_to_xyz(api, drop_x, drop_y, Z_SAFE)
        dobotArm.open_gripper(api)
        dobotArm.stop_pump(api)
        safe_move_to_xyz(api, drop_x, drop_y, Z_SAFE)

    print("\nBatch Complete.")
    return True

def phase_pick_place(target: Part):
    # This function can be used if you want to execute pick/place one by one with more complex logic in between, rather than in batches.
    while machine_state == "scanning target":
        pick_target = phase_detect_targets(target)
        if pick_target is not None:
            next_state()
            
    # --- PHASE 3 ACTIVATION LOCK ---
    # Camera must remain open during Phase 3 movement transits so safe_move_to_xyz can refresh the video and GUI window dynamically!
    
    while machine_state == "pick place":
        completed = phase_execute_batch(api, pick_target, drop_zone)
        if completed:
            next_state()
        else: break
        pass
 

# ---------------------------------------------------------
# MAIN EXECUTION
# contains an oversimplified state machine that runs the three phases sequentially. You can modify the logic to fit your specific use case.
# ---------------------------------------------------------
dobotArm.initialize_robot(api)
dobotArm.open_gripper(api)
dobotArm.stop_pump(api)
velcro  = Part(
        bounds=[
            np.array([[0,120,70], [10,255,255]]),   # lower red
            np.array([[170,120,70], [180,255,255]]), # upper red
        ],
        minContourArea=50
    )
purpleLegoBrick = Part(
        bounds=[
            np.array([[125,100,70], [165,255,255]])
        ],
        minContourArea=20
    )

while machine_state == "scanning plate":
    drop_zone = phase_detect_plates()
    if drop_zone is not None:
        next_state()

phase_pick_place(velcro)
machine_state = "scanning target"
phase_pick_place(purpleLegoBrick)

cap.release()
cv2.destroyAllWindows()

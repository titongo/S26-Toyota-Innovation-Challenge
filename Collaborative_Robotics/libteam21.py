import os

import cv2
import numpy as np

from cv2_enumerate_cameras import enumerate_cameras

VID_CAM_USB = 11205
VID_CAM_INTEGRATED = 1266
SATURATION_GAIN = float(os.getenv("SATURATION_GAIN", 1.35))
CAMERA_VID_BY_TYPE = {
    "usb": VID_CAM_USB,
    "integrated": VID_CAM_INTEGRATED,
}

def auto_select_camera(camera_type: str = "usb") -> tuple[int, int]:
    target_vid = CAMERA_VID_BY_TYPE.get(camera_type, VID_CAM_USB)
    cameras = enumerate_cameras()
    for cam in cameras:
        # print(f"Camera {cam.index}: {cam.name}, {cam.vid}, {cam.pid}, {cam.backend}")
        if cam.vid == target_vid:
            print(f"Selected Camera {cam.index}: {cam.name}")
            return cam.index, cam.backend
    return -1, -1


def boost_saturation(frame: np.ndarray, gain: float = SATURATION_GAIN) -> np.ndarray:
    if gain <= 1.0:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def main() -> None:
    cam_index, cam_backend = auto_select_camera("integrated")
    if cam_index == -1:
        print("No suitable camera found.")
        return

    print(f"Using camera index {cam_index} with backend {cam_backend}.")

    cam = cv2.VideoCapture(cam_index, cam_backend)
    
    while True:
        ret, frame = cam.read()
        frame = boost_saturation(frame)
        display_frame = frame.copy()

        cv2.imshow("Camera Feed", display_frame)
        if cv2.waitKey(1) == 27:  # ESC key
            break

if __name__ == "__main__":
    main()
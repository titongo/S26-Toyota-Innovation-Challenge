import cv2
from cv2_enumerate_cameras import enumerate_cameras

USB_CAM_VID = 11205

def autoSelectCamera():
    cameras = enumerate_cameras()
    for cam in cameras:
        # print(f"Camera {cam.index}: {cam.name}, {cam.vid}, {cam.pid}, {cam.backend}")
        if cam.vid == USB_CAM_VID:
            print(f"Selected Camera {cam.index}: {cam.name}")
            return (cam.index, cam.backend)

    return (-1, -1)


if __name__ == "__main__":
    cam_index, cam_backend = autoSelectCamera()
    if cam_index == -1:
        print("No suitable camera found.")
    else:
        print(f"Using camera index {cam_index} with backend {cam_backend}.")
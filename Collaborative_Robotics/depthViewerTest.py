# depthViewer.py
# Live depth video from the Astra Pro. Run this on its own:  python depthViewer.py
# If you see a coloured depth window, your depth driver is working (and you don't
# need to download anything else). Press ESC to quit.

import numpy as np
import cv2
from openni import openni2
from openni import _openni2 as c_api

OPENNI_REDIST = r"C:\Path\To\OpenNI2\Redist"   # <-- SAME path you put in the other files

openni2.initialize(OPENNI_REDIST)
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,   # values in millimetres
    resolutionX=640, resolutionY=480, fps=30))
depth_stream.start()

print("Depth stream running. ESC in the window to quit.")
while True:
    f = depth_stream.read_frame()
    depth = np.frombuffer(f.get_buffer_as_uint16(), np.uint16).reshape(f.height, f.width)  # mm

    # turn the raw mm values into something viewable: 0..2000 mm -> 0..255 -> colour
    # (lower the 2000 if the scene is close and the picture looks too dark/flat)
    vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255.0 / 2000.0), cv2.COLORMAP_JET)
    cv2.imshow("Astra Depth", vis)

    if cv2.waitKey(1) & 0xFF == 27:   # ESC
        break

depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()

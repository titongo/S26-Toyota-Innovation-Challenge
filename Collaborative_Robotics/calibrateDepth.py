# calibrateDepth.py
# Run this ONCE to create depth_calib.npz, which maps camera depth (mm) -> robot Z (mm).
# It uses two known surfaces: the bare table, and a block of known height on the table.
#
# Aim the Astra so the surface you are measuring sits under the CENTRE of the depth image.

import numpy as np
from openni import openni2
from openni import _openni2 as c_api

# ----------------- EDIT THESE -----------------
OPENNI_REDIST = r"C:\Users\husky\Downloads\Orbbec_OpenNI_v2.3.0.86-beta6_windows_release\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs"   # folder from the Orbbec OpenNI2 SDK
Z_TABLE = -64.0    # robot Z you read when the gripper tip TOUCHES the table (jog + read in DobotStudio)
BLOCK_H = 40.0     # measured height (mm) of a reference block you place on the table
# ----------------------------------------------

# Calibration pixel locations

TABLE_X = 250
TABLE_Y = 100

BLOCK_X = 375
BLOCK_Y = 100

DEPTH_W, DEPTH_H = 640, 480

openni2.initialize(OPENNI_REDIST)
dev = openni2.Device.open_any()
ds = dev.create_depth_stream()
ds.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
    resolutionX=DEPTH_W, resolutionY=DEPTH_H, fps=30))
ds.start()

def depth_at_point(x, y, k=6, frames=10):
    readings = []

    for _ in range(frames):
        f = ds.read_frame()
        img = np.frombuffer(
            f.get_buffer_as_uint16(),
            np.uint16
        ).reshape(f.height, f.width)

        patch = img[
            y-k:y+k+1,
            x-k:x+k+1
        ]

        vals = patch[patch > 0]

        if vals.size:
            readings.append(np.median(vals))

    return float(np.median(readings))
input(f"Clear table at pixel ({TABLE_X},{TABLE_Y}). Press Enter...")
d_table = depth_at_point(TABLE_X, TABLE_Y)
print(f"  table depth   = {d_table:.1f} mm")

input(f"Place the {BLOCK_H:.0f} mm block at pixel ({BLOCK_X},{BLOCK_Y}). Press Enter...")
d_block = depth_at_point(BLOCK_X, BLOCK_Y)
print(f"  block-top depth = {d_block:.1f} mm")

# Fit robot_Z = a*depth + b through the two reference points:
#   (d_table, Z_TABLE)  and  (d_block, Z_TABLE + BLOCK_H)
a = BLOCK_H / (d_block - d_table)     # negative: closer surface = smaller depth = higher Z
b = Z_TABLE - a * d_table
np.savez("depth_calib.npz", a=a, b=b)
print(f"\nSaved depth_calib.npz   a={a:.4f}  b={b:.2f}")
print("Sanity check: a should be near -1 if the camera looks straight down.")

ds.stop()
openni2.unload()

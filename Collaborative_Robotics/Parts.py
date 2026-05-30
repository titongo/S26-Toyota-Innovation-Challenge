import numpy as np
from typing import Sequence
import cv2
# Example instantiation:
# red_part = Part(
#     bounds=[
#         [[0, 120, 70],   [10, 255, 255]],   # lower red
#         [[170, 120, 70], [180, 255, 255]],   # upper red
#     ],
#     minContourArea=500
# )

#class for defining parts to be detected, with their color bounds and minimum contour area for detection
class Part:
    def __init__(self, bounds: Sequence[np.ndarray], minContourArea: int):
        self.bounds = np.asarray(bounds, dtype=object)
        self.minContourArea = minContourArea
    def returnMask(self):
        mask = None
        for bound in self.bounds:
            if mask is None:
                mask = cv2.inRange(hsv, np.array(bound[0]), np.array(bound[1]))
            else:
                mask += cv2.inRange(hsv, np.array(bound[0]), np.array(bound[1]))
        return mask
    
#cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255]))
#cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
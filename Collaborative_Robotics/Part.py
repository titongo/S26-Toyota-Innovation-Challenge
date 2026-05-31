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
    def __init__(self, bounds: Sequence[np.ndarray], minContourArea: int, frame=None):
        self.bounds = np.asarray(bounds)
        self.minContourArea = minContourArea
        if frame is not None:
            self.updateFrame(frame)
        else:
            self.frame = None
            self.hsv = None

    def updateFrame(self, frame):
        self.frame = frame
        self.hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (3,3), 0), cv2.COLOR_BGR2HSV)
    def returnMask(self):
        if self.hsv is None:
            raise ValueError("hsv not set — pass frame at construction or set self.hsv first")
        mask = None
        for bound in self.bounds:
            if mask is None:
                mask = cv2.inRange(self.hsv, np.array(bound[0]), np.array(bound[1]))
            else:
                mask += cv2.inRange(self.hsv, np.array(bound[0]), np.array(bound[1]))
        return mask
    
#cv2.inRange(self.hsv, np.array([0,120,70]), np.array([10,255,255]))
#cv2.inRange(self.hsv, np.array([170,120,70]), np.array([180,255,255]))
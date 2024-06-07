import numpy as np
import cv2

def ViewTransformer():
    def __init__(self):
        lane_width = 16
        lane_length = 19

        self.pixel_vertices = np.array(
            [652,523],
            [1151,482],
            [972,379],
            [545,413],
        )

        self.target_vertices = np.array({
            [0,lane_width],
            [lane_length,lane_width],
            [lane_length,0],
            [0,0],
        })

        self.pixel_vericies = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

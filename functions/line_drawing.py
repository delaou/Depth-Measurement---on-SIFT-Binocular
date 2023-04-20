import cv2
import numpy as np

def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    line_interval = 50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval*(k+1)), (2*width, line_interval*(k+1)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    return output
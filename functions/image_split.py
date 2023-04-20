import numpy as np
import cv2

def split_img(img, label_norm, width=640, height=640, gain=1, pad=0):
    cls = str(label_norm[0])
    x = float(label_norm[1]) * width
    y = float(label_norm[2]) * height
    w = int((float(label_norm[3]) * width) * gain + pad)
    h = int((float(label_norm[4]) * height) * gain + pad)

    x1 = int(x - w / 2.0)
    y1 = int(y - h / 2.0)
    x2 = int(x + w / 2.0)
    y2 = int(y + h / 2.0)
    
    place_arr = [x1, x2, y1, y2]

    split_img = img[y1: y2, x1: x2]
    
    return split_img, place_arr# split image and its original place information

def ImgsetLoader(img_dir, label_dir, type=0):
    label_norm = np.loadtxt(label_dir)
    label_mat = np.delete(label_mat, [i for i in range(label_mat.shape[0]) if label_mat[i][0] == 0], axis=0)
    label_norm = list(label_mat[label_mat.argmax(axis=0)[-1]])

    img = cv2.imread(img_dir, type)
    
    return img, label_norm
    
if __name__  == '__main__': 
    img_dir = r"D:\Filea\miceie\projects\Stereo_vision_lure\rectified\right\right0001.bmp"
    label_dir = r"D:\Filea\miceie\projects\Stereo_vision_lure\rectified\label_r\labels\right0001.txt"
    img, label_norm = ImgsetLoader(img_dir, label_dir)
    img, _ = split_img(img, label_norm, 2952, 1944)
    cv2.imshow('adasd', img)
    cv2.imwrite(r"D:\Filea\miceie\projects\Stereo_vision_lure\rectified\right1.bmp", img)
    cv2.waitKey()
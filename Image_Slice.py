import os
import numpy as np
import cv2

def split_img(img, label_norm, img_size, gain=1, pad=0):
    width = img_size[0]
    height = img_size[1]
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
    
    return split_img, place_arr

def ImgsetLoader(img_dir, label_dir, type=0):
    print(img_dir)
    print(label_dir)
    label_mat = np.loadtxt(label_dir)
    if len(label_mat.shape) > 1:
        label_mat = np.delete(label_mat, [i for i in range(label_mat.shape[0]) if label_mat[i][0] == 0], axis=0)
        label_norm = list(label_mat[label_mat.argmax(axis=0)[-1]])
    else:
        label_norm = list(label_mat)
        
    img = cv2.imread(img_dir, type)
    
    return img, label_norm



class img_label_pairset():
    
    def __init__(self, img_l_dir, label_l_dir, img_r_dir, label_r_dir, type=0):
        self.img_l, self.label_l = ImgsetLoader(img_l_dir, label_l_dir, type)
        self.img_size = self.img_l.shape[:2][::-1]
        self.img_r, self.label_r = ImgsetLoader(img_r_dir, label_r_dir, type)
    
    def __call__(self):
        pass
    
    def __getitem__(self):
        pass
    
    def __len__(self):
        pass
    
    def split(self):
        self.slice_l, self.slice_place_arr_l = split_img(self.img_l, self.label_l, self.img_size)
        self.slice_r, self.slice_place_arr_r = split_img(self.img_r, self.label_r, self.img_size)
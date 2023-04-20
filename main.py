import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import re
from Binocular_Camera import binocular_camera
from ultralytics import YOLO
from Image_Slice import img_label_pairset

import matplotlib
matplotlib.use('TkAgg')


def draw_line(img_l_dir, img_r_dir):
    img_l = cv2.imread(img_l_dir, 0)
    img_r = cv2.imread(img_r_dir, 0)
    
    height = max(img_l.shape[0], img_r.shape[0])
    width = img_l.shape[1] + img_r.shape[1]

    output = np.zeros((height, width), dtype=np.uint8)
    output[0:img_l.shape[0], 0:img_l.shape[1]] = img_l
    output[0:img_r.shape[0], img_l.shape[1]:] = img_r

    line_interval = 50
    
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval*(k+1)), (2*width, line_interval*(k+1)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    return output

def detect(img_root, save_label_root):
    model = YOLO(r'./test_on_1lure.pt')
    model.predict(source=img_root,
                  save=True, save_txt=True, save_conf=True, 
                  name=save_label_root)
    
def SIFT_match(imgA, imgB, type=0):
    sift = cv2.SIFT_create()
    kpsA, dpA = sift.detectAndCompute(imgA, None)
    kpsB, dpB = sift.detectAndCompute(imgB, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(dpA, dpB, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    
    ptA = []
    ptB = []
    for i, (m1, m2) in enumerate(matches):
        matchesMask[i] = [1, 0]
        pt1 = kpsA[m1.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
        pt2 = kpsB[m1.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
        ptA.append(pt1)
        ptB.append(pt2)
        # print(i, pt1, pt2)
        
    draw_params = dict(matchColor = (255, 0, 0),
        singlePointColor = (0, 0, 255),
        matchesMask = matchesMask,
        flags = 0)
    
    res = cv2.drawMatchesKnn(imgA, kpsA, imgB, kpsB, matches, None, **draw_params)

    cv2.imshow('1_vs_1_img', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ptA, ptB
          
if __name__ == '__main__': 
    img_l_root = r"F:\Filea\miceie\projects\Stereo_vision_lure\trial\example\left"
    img_r_root = r"F:\Filea\miceie\projects\Stereo_vision_lure\trial\example\right"
    save_root_l = r"F:\Filea\miceie\projects\Stereo_vision_lure\rectified\left"
    save_root_r = r"F:\Filea\miceie\projects\Stereo_vision_lure\rectified\right"
    save_label_root_l = r"F:\Filea\miceie\projects\Stereo_vision_lure\rectified\label_l"
    save_label_root_r = r"F:\Filea\miceie\projects\Stereo_vision_lure\rectified\label_r"
    
    camset = binocular_camera()
    camset(img_l_root=img_l_root, 
           img_r_root=img_r_root, 
           save_root_l=save_root_l, 
           save_root_r=save_root_r)
    
    # cv2.namedWindow('lined', 0)
    # cv2.resizeWindow('lined', 1080, 960)
    # cv2.imshow('lined', draw_line(r"D:\Filea\miceie\projects\Stereo_vision_lure\rectified\left\left0002.bmp", 
    #                               r"D:\Filea\miceie\projects\Stereo_vision_lure\rectified\right\right0002.bmp"))
    # cv2.waitKey()

    detect(save_root_l, save_label_root_l)
    detect(save_root_r, save_label_root_r)
    
    list = os.listdir(save_root_r)
    list = [i[-8:-4] for i in list]
    

    for index, ele in enumerate(list):
        img_l_dir = os.path.join(save_root_l, f'left{ele}.bmp')
        label_l_dir = os.path.join(save_label_root_l, 'labels', f'left{ele}.txt')
        img_r_dir = os.path.join(save_root_r, f'right{ele}.bmp')
        label_r_dir = os.path.join(save_label_root_r, 'labels', f'right{ele}.txt')
        
        sete = img_label_pairset(img_l_dir, label_l_dir, img_r_dir, label_r_dir)
        sete.split()
        points_l0, points_r0 = SIFT_match(sete.slice_l, sete.slice_r)
        points_l0 = [[pt[0]+sete.slice_place_arr_l[0], camset.img_size[1]-sete.slice_place_arr_l[3]+pt[1]] for pt in points_l0]
        points_r0 = [[pt[0]+sete.slice_place_arr_r[0], camset.img_size[1]-sete.slice_place_arr_r[3]+pt[1]] for pt in points_r0]
        points_l = []
        points_r = []
        for point_l, point_r in zip(points_l0, points_r0):
            if abs(point_l[1]-point_r[1]) < 5:
                points_l.append(point_l)
                points_r.append(point_r)
            else:
                pass
        # print(points_l)
        # print(points_r)
        points_3d = camset.depth_calc(points_l, points_r)
        # print(points_3d[1203][830])
        depth_img = points_3d.T[2]
        mask = (depth_img > 0)*(depth_img < 10000)
        x_index, y_index = np.where(mask)
        x = np.array([points_3d.T[0][x][y] for x, y in zip(x_index, y_index)])
        y = np.array([points_3d.T[1][x][y] for x, y in zip(x_index, y_index)])
        z = np.array([depth_img[x][y] for x, y in zip(x_index, y_index)])
        print(x)
        print(y)
        print(z)
        fig = plt.figure()
        ax = fig.add_axes(Axes3D(fig))
        ax.scatter(x, y, z, c='r')
        plt.show()
            
        

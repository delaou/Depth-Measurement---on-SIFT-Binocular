import numpy as np
import cv2
import glob
from line_drawing import draw_line

def Calibrate(sampt,adset,limnum=30):
    #单目标定
    img_adset=glob.glob(adset)
    sterpt_arr=[]
    cornpt_arr=[]
    fal_read=[]
    count=0
    sterpt=np.zeros((sampt[0]*sampt[1],3),np.float32)
    x,y=np.mgrid[0:sampt[0],0:sampt[1]]
    sterpt[:,:2]=cv2.merge((np.transpose(x),np.transpose(y))).reshape(-1,2)#生成世界坐标
    for img_ad in img_adset:
        img=cv2.imread(img_ad,0)#依次读入图片
        ret,cornpt=cv2.findChessboardCorners(img,sampt)#寻找棋盘点
        if ret==True:
            cornpt=cv2.cornerSubPix(img,cornpt,(11,11),(-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001))#亚像素角点检测
            sterpt_arr.append(sterpt)
            cornpt_arr.append(cornpt)
            count+=1
            if count>=limnum:
                break
        else:
            fal_read.append(img_ad)#记录未找到棋盘点的图片
    img_size=img.shape[::-1]
    ret,camMat,dist,rvecs,tvecs=cv2.calibrateCamera(sterpt_arr,cornpt_arr,img_size,None,None)#相机标定，计算参数
    return sterpt_arr,cornpt_arr,camMat,dist


def RectifyMaps(binocular_camera):
    rec_rvecs_l, rec_rvecs_r, rec_tvecs_l, rec_tvecs_r, Q, validPixROI1,validPixROI2 = \
        cv2.stereoRectify(binocular_camera.camMat_l, binocular_camera.dist_l, binocular_camera.camMat_r, binocular_camera.dist_r,
                          binocular_camera.img_size[:2], binocular_camera.rvecs, binocular_camera.tvecs, alpha=0)
        
    l_map1, l_map2 = cv2.initUndistortRectifyMap(binocular_camera.camMat_l, binocular_camera.dist_l, rec_rvecs_l, rec_tvecs_l,
                                              binocular_camera.img_size[:2], cv2.CV_16SC2)
    r_map1, r_map2 = cv2.initUndistortRectifyMap(binocular_camera.camMat_r, binocular_camera.dist_r, rec_rvecs_r, rec_tvecs_r,
                                              binocular_camera.img_size[:2], cv2.CV_16SC2)
    
    return l_map1, l_map2, r_map1, r_map2, Q


def Image_Rectify(binocular_camera, img_l_dir, img_r_dir):
    img_l = cv2.imread(img_l_dir, binocular_camera.l_type)
    img_r = cv2.imread(img_r_dir, binocular_camera.r_type)

    img_l = cv2.undistort(img_l, binocular_camera.camMat_l, binocular_camera.dist_l)
    img_r = cv2.undistort(img_r, binocular_camera.camMat_r, binocular_camera.dist_r)

    rectified_img_l = cv2.remap(img_l, binocular_camera.l_map1, binocular_camera.l_map2, cv2.INTER_AREA)
    rectified_img_r = cv2.remap(img_r, binocular_camera.r_map1, binocular_camera.r_map2, cv2.INTER_AREA)

    return rectified_img_l, rectified_img_r
import numpy as np
import cv2
import os

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

def depth(binocular_camera, points_l, points_r):
    disp_img = np.zeros(binocular_camera.img_size, dtype=np.float32)
    for point_l, point_r in zip(points_l, points_r):
        disp = point_l[0]-point_r[0]
        disp_img[points_l] = disp
    points_3d = cv2.reprojectImageTo3D(disp_img, binocular_camera.Q, handleMissingValues=False)
    return points_3d


class binocular_camera():
    
    def __init__(self):
        
        self.camMat_l = np.array([[7.813007696907518e+03, 0, 0], 
                                  [0, 7.768873078657097e+03, 0], 
                                  [1.991969896930244e+02, 1.218487412399986e+03, 1]]).T
        self.camMat_r = np.array([[5.855949046226611e+03, 0, 0], 
                                  [0, 5.603299054119371e+03, 0], 
                                  [3.910062926540840e+02, 2.122672494051519e+03, 1]]).T
    
        self.dist_l = np.array([0.167755704935931, 0.096637885335477, 0, 0, 0])
        self.dist_r = np.array([0.237169573179642, -1.396779961405439, 0, 0, 0])
        
        self.rvecs = np.array([[0.999957639036066, 0.008187078431089, -0.004206171677414], 
                               [-0.007349825664001, 0.985320736635736, 0.170555052749312], 
                               [0.005540775769283, -0.170516913244332, 0.985339171098714]]).T
        
        self.tvecs = np.array([-78.699146029827360, -6.595772554827722, 72.028101691905580]).T
        
        self.img_size = (2952, 1944)
        
        self.l_type = 0
        self.r_type = 0

    def Get_Rectify_Maps(self):
        self.l_map1, self.l_map2, self.r_map1, self.r_map2, self.Q = RectifyMaps(self)
        
    def __call__(self, img_l_root, img_r_root, save_root):
        self.Get_Rectify_Maps()
        for index, img_l_name, img_r_name in enumerate(zip(os.listdir(img_l_root), os.listdir(img_r_root))):
            img_l_dir = os.path.join(img_l_root, img_l_name)
            img_r_dir = os.path.join(img_r_root, img_r_name)
            
            img_l, img_r = Image_Rectify(self, img_l_dir, img_r_dir)
            
            cv2.imwrite(os.path.join(save_root, 'left', img_l_name), img_l)
            cv2.imwrite(os.path.join(save_root, 'right', img_r_name), img_r)
        print(f'Rectify Finished: {index+1} pairs have been loaded')
    
    def __getitem__(self, param):
        pass
    
    def depth_calc(self, points_l, points_r):
        points_3d = depth(self, points_l, points_r)
        return points_3d
    
if __name__ == '__main__': 
    pass
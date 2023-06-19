import numpy as np
import cv2
import os
from functions.StereoCalibration import stereocalibrate


class binocular_camera(object): 
    
    def __init__(self, calibrate=False, num_point=None, root_l=None, root_r=None): 
        
        if calibrate == True: 
            self.camMat_l, self.dist_l, self.camMat_r, self.dist_r, self.R, self.T, self.img_size = \
                stereocalibrate(num_point, root_l, root_r)
            print('Calibration finished. ')
            self.l_type = 0
            self.r_type = 0

        else: 
            self.camMat_l = np.array([[5.232854530882389e+03, 0, 0], 
                                    [0, 5.231716770525062e+03, 0], 
                                    [1.282357665861073e+03, 9.146482864888435e+02, 1]]).T
            self.camMat_r = np.array([[5.116792385007132e+03, 0, 0], 
                                    [0, 5.117062580465808e+03, 0], 
                                    [1.320276691136698e+03, 1.100639471120834e+03, 1]]).T
        
            self.dist_l = np.array([-0.104813076347317, 0.129459331763937, 0, 0, 0])
            self.dist_r = np.array([-0.055810618566240, -0.182687128182945, 0, 0, 0])
            
            self.R = np.array([[-0.999075407292026, 0.019469219487619, 0.038331188826022], 
                                [-0.019021635970218, -0.999746972038066, 0.012007050667640], 
                                [0.038555257868282, 0.011266827115994, 0.999192949683618]]).T
            
            self.T = np.array([-0.262149346032060, -37.438824849953410, -1.251828836237686]).T
            
            self.img_size = (2952, 1944)
            
            self.l_type = 0
            self.r_type = 0

    def Get_Rectify_Maps(self): 
        
        self.re_R_l, self.re_R_r, self.re_T_l, self.re_T_r, self.Q, validPixROI1, validPixROI2 = \
            cv2.stereoRectify(self.camMat_l, self.dist_l, self.camMat_r, self.dist_r,
                          self.img_size[:2], self.R, self.T, 1, (0, 0))
        
        self.l_map1, self.l_map2 = cv2.initUndistortRectifyMap(self.camMat_l, self.dist_l, self.re_R_l, self.re_T_l,
                                              self.img_size[:2], cv2.CV_16SC2)
        self.r_map1, self.r_map2 = cv2.initUndistortRectifyMap(self.camMat_r, self.dist_r, self.re_R_r, self.re_T_r,
                                              self.img_size[:2], cv2.CV_16SC2)    
        
    def __call__(self, img_l_root, img_r_root, save_root_l, save_root_r): 
        
        self.Get_Rectify_Maps()
        
        for index, (img_l_name, img_r_name) in enumerate(zip(os.listdir(img_l_root), os.listdir(img_r_root))):
            img_l_dir = os.path.join(img_l_root, img_l_name)
            img_r_dir = os.path.join(img_r_root, img_r_name)
            
            img_l = cv2.imread(img_l_dir, self.l_type)
            img_r = cv2.imread(img_r_dir, self.r_type)

            img_l = cv2.undistort(img_l, self.camMat_l, self.dist_l)
            img_r = cv2.undistort(img_r, self.camMat_r, self.dist_r)

            self.rectified_img_l = cv2.remap(img_l, self.l_map1, self.l_map2, cv2.INTER_AREA)
            self.rectified_img_r = cv2.remap(img_r, self.r_map1, self.r_map2, cv2.INTER_AREA)
            
            cv2.imwrite(os.path.join(save_root_l, img_l_name), self.rectified_img_l)
            cv2.imwrite(os.path.join(save_root_r, img_r_name), self.rectified_img_r)
            
        print(f'Rectify Finished: {index+1} pairs have been loaded')
        
    def __getitem__(self, param):
        pass
    
    def depth_calc(self, points_l, points_r): 
        
        disp_img = np.zeros(self.img_size, dtype=np.float32)
        for point_l, point_r in zip(points_l, points_r):
            disp = point_l[0]-point_r[0]
            disp_img[int(point_l[1])][int(point_l[0])] = disp
        points_3d = cv2.reprojectImageTo3D(disp_img, self.Q, handleMissingValues=True)
        return points_3d
    
if __name__ == '__main__': 
    cas = binocular_camera()
    cas(img_l_root="", 
        img_r_root="", 
        save_root_l="", 
        save_root_r="")
    print(cas.Q)

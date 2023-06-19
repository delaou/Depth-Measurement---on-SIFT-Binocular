import numpy as np
import cv2
import os



def calibrate(num_point, root, limnum=100): 
    # calibrate the single camera
    
    name_imgs = os.listdir(root)
    world_coordinates_list = []
    camera_coordinates_list = []
    fail_read = []
    count = 0
    world_coordinates = np.zeros((num_point[0] * num_point[1], 3), dtype=np.float32)
    x, y = np.mgrid[0:num_point[0], 0:num_point[1]]
    world_coordinates[:, :2] = cv2.merge((x.T, y.T)).reshape(-1, 2)
    
    for name_img in name_imgs: 
        dir_img = os.path.join(root, name_img)
        img = cv2.imread(dir_img, 0)
        print(f'Searching corners of {name_img}. ')
        ret, camera_coordinates = cv2.findChessboardCorners(img, num_point)
        
        if ret == True: 
            camera_coordinates = cv2.cornerSubPix(img, camera_coordinates, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            print('Successfully found. \n')
            world_coordinates_list.append(world_coordinates)
            camera_coordinates_list.append(camera_coordinates)
            count += 1
            
            if count >= limnum:
                break
            
        else: 
            print('Fail to find. \n')
            fail_read.append(dir_img)
            
    img_size = img.shape[::-1]        
    ret, camMat, dist, R, T = cv2.calibrateCamera(world_coordinates_list, camera_coordinates_list, img_size, None, None)
    
    return world_coordinates_list, camera_coordinates_list, camMat, dist, R, T, img_size



def stereocalibrate(num_point, root_l, root_r, limnum=100): 
    
    world_coordinates_list, camera_coordinates_list_l, camMat_l, dist_l, R_l, T_l, img_size =\
        calibrate(num_point, root_l, limnum)
    _, camera_coordinates_list_r, camMat_r, dist_r, R_r, T_r, _ = \
        calibrate(num_point, root_r, limnum)
        
    ret, camMat_l, dist_l, camMat_r, dist_r, R, T, E, F = \
        cv2.stereoCalibrate(world_coordinates_list, camera_coordinates_list_l, camera_coordinates_list_r, camMat_l, dist_l, camMat_r, dist_r, img_size)
    
    return camMat_l, dist_l, camMat_r, dist_r, R, T, img_size
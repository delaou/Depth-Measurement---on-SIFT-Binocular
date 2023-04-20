import cv2

def ORB_match(imgA_dir, imgB_dir, type=0):
    imgA = cv2.imread(imgA_dir, type)
    imgB = cv2.imread(imgB_dir, type)

    orb = cv2.ORB_create()
    kpsA, dpA = orb.detectAndCompute(imgA,None)
    kpsB, dpB = orb.detectAndCompute(imgB,None)

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
        print(i, pt1, pt2)
        
    draw_params = dict(matchColor = (255, 0, 0),
        singlePointColor = (0, 0, 255),
        matchesMask = matchesMask,
        flags = 0)
    
    res = cv2.drawMatchesKnn(imgA, kpsA, imgB, kpsB, matches, None, **draw_params)

    cv2.imshow('1_vs_1_img', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ptA, ptB

def SIFT_match(imgA_dir, imgB_dir, type=0):
    imgA = cv2.imread(imgA_dir, type)
    imgB = cv2.imread(imgB_dir, type)

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
        print(i, pt1, pt2)
        
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
    ptA, ptB = SIFT_match(r"D:\Filea\miceie\projects\Stereo_vision_lure\rectified\left1.bmp", r"D:\Filea\miceie\projects\Stereo_vision_lure\rectified\right1.bmp")
    ptA, ptB
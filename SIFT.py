import time

import pysift
from SiftFuncs import *


def generateSIFT(I):
    I = I.astype('float32')
    # I = cv2.resize(I, (512, 512))
    I0 = cv2.pyrUp(I)
    OctaveNum = CalcPyrNum([I.shape[0], I.shape[1]])
    scale = 2
    s = 3
    sigma = 1.6
    # [pyrPics, As, Bs, DoGs] = GenerateGausPyrPics(I0, OctaveNum, scale=scale, s=3, gaussKernelSize=3, sigma=1.414213)
    """生成不同图像尺寸的高斯金字塔,每个图像尺寸(Octave)有s个不同的高斯尺度(Intervals)"""
    GaussPyrPics = GenerateGausPyrPics(I0, OctaveNum, scale=scale, s=s, sigma=sigma)
    # ShowPyrPics3("Gaussian Pyr", GaussPyrPics)
    """根据高斯金字塔,做出高斯差分金字塔,结构类似,但Intervals减小1"""
    DoGs = GenerateDoGImages(GaussPyrPics)
    # ShowPyrPics3("DoGs Pyr", DoGs)
    """根据高斯差分金字塔,每个图像尺寸层的接连三个Interval上遍历出中间层的极大值点,共s-3"""
    start = time.process_time()
    KeyPoints = FindMaxMin(GaussPyrPics, DoGs, s, sigma, ImBoarderWidth=5, ContrastThreshold=0.04)
    # KeyPoints = pysift.findScaleSpaceExtrema(GaussPyrPics, DoGs, s, sigma)
    eclips = time.process_time() - start
    print("Find Points and Cal costed " + str(eclips))
    """"""
    keypoints = pysift.removeDuplicateKeypoints(KeyPoints)
    keypoints = pysift.convertKeypointsToInputImageSize(keypoints)
    start = time.process_time()
    descriptors = pysift.generateDescriptors(keypoints, GaussPyrPics)
    eclips = time.process_time() - start
    print("Generate Discriptor cost : " + str(eclips))
    return keypoints, descriptors


I1 = cv2.imread("box.png")
I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
I1 = cv2.resize(I1, (int(I1.shape[1] / 2), int(I1.shape[0] / 1.5)))
I2 = cv2.imread("box_in_scene.png")
I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
I2 = cv2.resize(I2, (int(I2.shape[1] / 2), int(I2.shape[0] / 2)))
kp1, des1 = generateSIFT(I1)
kp2, des2 = generateSIFT(I2)
print("特征生成完毕")
# Initialize and use FLANN

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    # Estimate homography between template and scene
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

    # Draw detected template in scene image
    [h, w] = [I1.shape[0],I1.shape[1]]
    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    I2 = cv2.polylines(I2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    [h1, w1] = [I1.shape[0],I1.shape[1]]
    [h2, w2] = [I2.shape[0],I2.shape[1]]
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = I1
        newimg[:h2, w1:w1 + w2, i] = I2

    # Draw SIFT keypoint matches
    for m in good:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        cv2.line(newimg, pt1, pt2, (255, 0, 0))

    plt.imshow(newimg)
    plt.show()
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

"""
AbsDoGs = np.abs(DoGs)
MaxMinFlag = []
for i in range(OctaveNum):
    MaxMinFlag.append(FindMaximaMinima(DoGs[i]))
UniSize = []
UniSize.append(MaxMinFlag[0])
Eliminated = MaxMinFlag[0]
for i in range(1, OctaveNum):
    UniSize.append(cv2.resize(MaxMinFlag[i], (MaxMinFlag[0].shape[1], MaxMinFlag[0].shape[0]), cv2.INTER_NEAREST))
    UniSize[i] = np.where(UniSize[i] == 0, 0, 1.0)
    Eliminated = Eliminated * UniSize[i]
ShowPicPyr("Unisize", UniSize)
Myimshow("Eliminated", Eliminated)
MaxMinFlagEli = EliminateScaleNonKeys(MaxMinFlag.copy(), CurrentOctave=0, scale=scale)
ShowPicPyr("Flag ", MaxMinFlag)
ShowPicPyr("Flag ", MaxMinFlagEli)
GI = Direct2D_GaussianBlur(I, 3, 1.414)
I1d = Double1D_GaussianBlur(I, 3, 1.414)
IdiffGaussian = np.abs(DiffGaussian(I, 3, 3, sigma1=1.414213, sigma2=1.414213))
Myimshow("Original", I)
Myimshow("Difference", IdiffGaussian)
cv2.waitKey(0)
"""

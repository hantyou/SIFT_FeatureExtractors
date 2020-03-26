from SiftFuncs import *

I = cv2.imread("box.png")
# I = cv2.resize(I, (512, 512))
gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY) / 255
I = gray
I0 = cv2.pyrUp(I)
OctaveNum = CalcPyrNum([I.shape[0], I.shape[1]])
scale = 1.5
s = 3
sigma = 1.414213
# [pyrPics, As, Bs, DoGs] = GenerateGausPyrPics(I0, OctaveNum, scale=scale, s=3, gaussKernelSize=3, sigma=1.414213)
"""生成不同图像尺寸的高斯金字塔,每个图像尺寸(Octave)有s个不同的高斯尺度(Intervals)"""
GaussPyrPics = GenerateGausPyrPics(I0, OctaveNum, scale=scale, s=s, sigma=sigma)
ShowPyrPics3("Gaussian Pyr", GaussPyrPics)
"""根据高斯金字塔,做出高斯差分金字塔,结构类似,但Intervals减小1"""
DoGs = GenerateDoGImages(GaussPyrPics)
ShowPyrPics3("DoGs Pyr", DoGs)
"""根据高斯差分金字塔,每个图像尺寸层的接连三个Interval上遍历出中间层的极大值点,共s-3"""
MMLocations = FindMaxMin(DoGs, s, sigma, ImBoarderWidth=3, ContrastThresh=0.04)
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

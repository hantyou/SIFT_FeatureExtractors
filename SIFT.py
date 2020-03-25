from SiftFuncs import *

I = cv2.imread("box.png")
# I = cv2.resize(I, (512, 512))
gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I = I / 255
gray = gray / 255
I = gray
I0 = cv2.pyrUp(I)
pyrPics = []
pyrPics.append(I0)
OctaveNum = CalcPyrNum([I.shape[0], I.shape[1]])
scale = 1.5
[pyrPics, As, Bs, DoGs] = GenerateGausPyrPics(I0, OctaveNum, scale=scale)
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

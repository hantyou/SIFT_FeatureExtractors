from math import log

import cv2
import numpy as np


def CalcPyrNum(shape):
    """Compute number of octaves in image pyramid as function of base image shape (OpenCV default)
    """
    return int(round(log(min(shape)) / log(2) - 3))


def GeneratePyrPics(I, LayerNum):
    pyrPics = []
    I0 = cv2.pyrUp(I)
    pyrPics.append(I0)
    pyrPics.append(I)
    for i in range(2, LayerNum):
        pyrPics.append(MyBiLiResize(pyrPics[i - 1], 1.5))
        # pyrPics.append(cv2.pyrDown(pyrPics[i - 1]))
    return pyrPics


def GenerateGausPyrPics(I, LayerNum, scale=1.5):
    pyrPics = []
    As = []
    Bs = []
    DoGs = []
    LayerNum += 1
    for i in range(LayerNum):

        tempA = Double1D_GaussianBlur(I, 7, 1.414213)
        tempB = Double1D_GaussianBlur(tempA, 7, 1.414213)
        pyrPics.append(I)
        As.append(tempA)
        Bs.append(tempB)
        DoGs.append(tempA - tempB)
        # I = MyBiLiResize(Bs[i], scale)
        I = cv2.resize(Bs[i], (int(Bs[i].shape[0] / scale), int(Bs[i].shape[1] / scale)))
        if i == 0:
            I = cv2.resize(Bs[i], (int(Bs[i].shape[0] / 2), int(Bs[i].shape[1] / 2)))

        # I = cv2.pyrDown(Bs[i], (Bs[i].shape[0] / 1.5, Bs[i].shape[0] / 1.5))
    return [pyrPics, As, Bs, DoGs]


def ShowPicPyr(string, pyrPics):
    len = pyrPics.__len__()
    for i in range(len):
        cv2.namedWindow(string + "Pic" + str(i), cv2.WINDOW_NORMAL)
        cv2.imshow(string + "Pic" + str(i), pyrPics[i])
        print(pyrPics[i].shape)


def FindMaximaMinima(I):
    """此处在循环中使用函数调用严重影响算法速度
    def Judge8(I, j, i):
        tp = I[j - 1:j + 2, i - 1:i + 2]
        maxima = np.max(tp)
        minima = np.min(tp)
        if maxima == tp[1,1] or minima == tp[1,1]:
            return 1
        else:
            return 0
    """
    [h, w] = [I.shape[0], I.shape[1]]
    maxout = np.zeros((h, w))
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            tp = I[j - 1:j + 2, i - 1:i + 2]
            maxima = np.max(tp)
            minima = np.min(tp)
            if maxima == tp[1, 1] or minima == tp[1, 1]:
                flag = 1
            else:
                flag = 0
            maxout[j, i] = flag
    return maxout


def EliminateScaleNonKeys(MaxMinFlag, CurrentOctave=0, scale=1.5):
    def DeepSearchEli(MaxMinFlag, y, x, o, scale):
        len = MaxMinFlag.__len__()
        [cy, cx] = [MaxMinFlag[o].shape[0], MaxMinFlag[o].shape[1]]
        y1 = int(y / scale)
        x1 = int(x / scale)
        if o == 1:
            y1 = int(y / 2)
            x1 = int(x / 2)
        if CurrentOctave == len:
            """
            if MaxMinFlag[o][y1, x1] == 0:
                return MaxMinFlag[o][y1, x1]
            else:
                return 1
            """
            return MaxMinFlag[o][y1, x1]
        else:
            if MaxMinFlag[o][y1, x1] == 1:
                MaxMinFlag[o][y1, x1] = DeepSearchEli(MaxMinFlag, y1, x1, o + 1, scale)
            else:
                MaxMinFlag[o][y1, x1] = 0
            return MaxMinFlag[o][y1, x1]

    len = MaxMinFlag.__len__()
    yxs = np.where(MaxMinFlag[CurrentOctave] == 1)
    ys = yxs[0]
    xs = yxs[1]
    n = ys.__len__()
    for k in range(n):
        y = ys[k]
        x = xs[k]
        MaxMinFlag[CurrentOctave][y, x] = DeepSearchEli(MaxMinFlag, y, x, CurrentOctave + 1, scale)
    return MaxMinFlag


def Direct2D_GaussianBlur(I, SizeOfGaussianKernel=3, sigma=1.414213):
    if not SizeOfGaussianKernel % 2:
        print("需要高斯核大小为奇数")
        quit()
    g1 = cv2.getGaussianKernel(SizeOfGaussianKernel, sigma)
    g2 = cv2.getGaussianKernel(SizeOfGaussianKernel, sigma)
    k = g1 * g2.T
    out = cv2.filter2D(I, -1, kernel=k)
    return out


def Double1D_GaussianBlur(I, SizeOfGaussianKernel=3, sigma=1.414213):
    if not SizeOfGaussianKernel % 2:
        print("需要高斯核大小为奇数")
        quit()
    g1 = cv2.getGaussianKernel(SizeOfGaussianKernel, sigma)
    g2 = cv2.getGaussianKernel(SizeOfGaussianKernel, sigma)
    out = cv2.filter2D(I, -1, kernel=g1)
    out = cv2.filter2D(out, -1, kernel=g2.T)
    return out


def Single1D_GaussianBlur(I, SizeOfGaussianKernel=3, sigma=1.414213, T_orNot=0):
    if not SizeOfGaussianKernel % 2:
        print("需要高斯核大小为奇数")
        quit()
    g1 = cv2.getGaussianKernel(SizeOfGaussianKernel, sigma)
    if T_orNot == 0:
        out = cv2.filter2D(I, -1, kernel=g1)
    else:
        out = cv2.filter2D(I, -1, kernel=g1.T)
    return out


def DiffGaussian(I, kernels1, kernels2, sigma1=1.414213, sigma2=1.414213):
    A = Single1D_GaussianBlur(I, kernels1, sigma=sigma1, T_orNot=0)
    B = Single1D_GaussianBlur(I, kernels2, sigma=sigma2, T_orNot=1)
    out = A - B
    return out


def MyResize(I, factor):
    w = I.shape[0]
    h = I.shape[1]
    w0 = int(w / factor)
    h0 = int(h / factor)
    out = cv2.resize(I, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return out


def Myimshow(str, I):
    cv2.namedWindow(str, cv2.WINDOW_NORMAL)
    cv2.imshow(str, I)


def MyBiLiResize(I, factor):
    def InsideBiLiResize(I, factor):
        w0 = I.shape[1]
        h0 = I.shape[0]
        w = int(w0 / factor)
        h = int(h0 / factor)
        out = np.zeros(shape=(h, w))
        for i in range(w):
            for j in range(h):
                i0 = i * factor
                j0 = j * factor
                x1 = int(i0)
                y1 = int(j0)
                x2 = x1 + 1
                y2 = y1
                x3 = x1
                y3 = y1 + 1
                x4 = x2
                y4 = y3
                x_a = i0 - x1
                x_b = 1 - x_a
                y_a = j0 - y1
                y_b = 1 - y_a
                v1 = I[y1, x1]
                v2 = I[y2, x2]
                v3 = I[y3, x3]
                v4 = I[y4, x4]
                tpv1 = x_b * v1 + x_a * v2
                tpv2 = x_b * v3 + x_a * v4
                out[j, i] = y_b * tpv1 + y_a * tpv2
        return out

    if factor <= 1:
        print("只能输入大于1的系数")
        quit()
    w0 = I.shape[1]
    h0 = I.shape[0]
    w = int(w0 / factor)
    h = int(h0 / factor)
    chn = int(I.size / w0 / h0)
    if chn > 1:
        out = np.zeros(shape=(h, w, chn))
        for c in range(chn):
            out[:, :, c] = InsideBiLiResize(I[:, :, c], factor)
    elif chn == 1:
        out = np.zeros(shape=(h, w))
        out = InsideBiLiResize(I, factor)
    return out


I = cv2.imread("28.jpg")
I = cv2.resize(I, (512, 512))
gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I = I / 255
gray = gray / 255
I = gray
I0 = cv2.pyrUp(I)
# I = cv2.resize(I, (2048, 2048))
pyrPics = []
pyrPics.append(I)
OctaveNum = CalcPyrNum([I.shape[0], I.shape[1]])
scale = 1.5
[pyrPics, As, Bs, DoGs] = GenerateGausPyrPics(I0, OctaveNum, scale=scale)
# pyrPics = GeneratePyrPics(I, OctaveNum)
# ShowPicPyr("pyr", pyrPics)
AbsDoGs = np.abs(DoGs)
ShowPicPyr("AbsDoGs", AbsDoGs)
ShowPicPyr("pyrPics", pyrPics)
cv2.waitKey(0)
MaxMinFlag = []
for i in range(OctaveNum):
    MaxMinFlag.append(FindMaximaMinima(AbsDoGs[i]))
MaxMinFlagEli = EliminateScaleNonKeys(MaxMinFlag.copy(), CurrentOctave=0, scale=scale)
ShowPicPyr("Flag Eliminated Scale wise unmatchable ", MaxMinFlagEli)
# ShowPicPyr("DoGs ", DoGs)
ShowPicPyr("Flag ", MaxMinFlag)
GI = Direct2D_GaussianBlur(I, 7, 1.414)
I1d = Double1D_GaussianBlur(I, 7, 1.414)
# myresizetest = MyBiLiResize(I1d, 2)
# sysresize = MyResize(I1d, 2)
IdiffGaussian = np.abs(DiffGaussian(I, 7, 7, sigma1=1.414213, sigma2=1.414213))
Myimshow("Original", I)
Myimshow("Difference", IdiffGaussian)
"""
Myimshow("My resize function result", myresizetest)
Myimshow("System resize function result", sysresize)
"""
cv2.waitKey(0)

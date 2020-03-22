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
        pyrPics.append(cv2.pyrDown(pyrPics[i - 1]))
    return pyrPics


def GenerateGausPyrPics(I, LayerNum):
    pyrPics = []
    As = []
    Bs = []
    LayerNum += 1
    for i in range(LayerNum):
        tempA = Double1D_GaussianBlur(I, 7, 1.414213)
        tempB = Double1D_GaussianBlur(tempA, 7, 1.414213)
        pyrPics.append(I)
        As.append(tempA)
        Bs.append(tempB)
        I = cv2.pyrDown(Bs[i])
    return [pyrPics, As, Bs]


def ShowPicPyr(pyrPics):
    len = pyrPics.__len__()
    for i in range(len):
        cv2.namedWindow("Pic" + str(i), cv2.WINDOW_NORMAL)
        cv2.imshow("Pic" + str(i), pyrPics[i])
        print(pyrPics[i].shape)


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
gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I = I / 255
gray = gray / 255
I0 = cv2.pyrUp(I)
# I = cv2.resize(I, (2048, 2048))
pyrPics = []
pyrPics.append(I)
OctaveNum = CalcPyrNum([I.shape[0], I.shape[1]])
[pyrPics, As, Bs] = GenerateGausPyrPics(I0, OctaveNum)
# pyrPics = GeneratePyrPics(I, OctaveNum)
ShowPicPyr(pyrPics)
GI = Direct2D_GaussianBlur(I, 7, 1.414)
I1d = Double1D_GaussianBlur(I, 7, 1.414)
# myresizetest = MyBiLiResize(I1d, 2)
# sysresize = MyResize(I1d, 2)
IdiffGaussian = DiffGaussian(I, 7, 7, sigma1=1.414213, sigma2=1.414213)
Myimshow("Original", I)
Myimshow("Difference", IdiffGaussian)
"""
Myimshow("My resize function result", myresizetest)
Myimshow("System resize function result", sysresize)
"""
cv2.waitKey(0)

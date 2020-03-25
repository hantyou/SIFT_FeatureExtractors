from math import log
from ImageFuncs import *
from ShowFuncs import *

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


def GenerateGausPyrPics(I, LayerNum, scale=1.5, gaussKernelSize=3):
    pyrPics = []
    As = []
    Bs = []
    DoGs = []
    LayerNum += 1
    for i in range(LayerNum):

        tempA = Double1D_GaussianBlur(I, gaussKernelSize, 1.414213)
        tempB = Double1D_GaussianBlur(tempA, gaussKernelSize, 1.414213)
        pyrPics.append(I)
        As.append(tempA)
        Bs.append(tempB)
        DoGs.append(tempA - tempB)
        I = MyBiLiResize(Bs[i], scale)
        # I = cv2.resize(Bs[i], (int(Bs[i].shape[1] / scale), int(Bs[i].shape[0] / scale)))
        if i == 0:
            I = MyBiLiResize(Bs[i], 2)
            # I = cv2.resize(Bs[i], (int(Bs[i].shape[1] / 2), int(Bs[i].shape[0] / 2)))

        # I = cv2.pyrDown(Bs[i], (Bs[i].shape[0] / 1.5, Bs[i].shape[0] / 1.5))
    return [pyrPics, As, Bs, DoGs]


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


def DecideKeys(I):
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
        # print(o)
        len = MaxMinFlag.__len__()
        y1 = int(y / scale)
        x1 = int(x / scale)
        if o == 1:
            y1 = int(y / 2)
            x1 = int(x / 2)
        if o == len - 1:
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
        # print((y,x))
        MaxMinFlag[CurrentOctave][y, x] = DeepSearchEli(MaxMinFlag, y, x, CurrentOctave + 1, scale)
        cv2.imshow("Realtime MaxMinFlag0", MaxMinFlag[0])
        cv2.imshow("Realtime MaxMinFlag1", MaxMinFlag[1])
        cv2.imshow("Realtime MaxMinFlag2", MaxMinFlag[2])
        cv2.imshow("Realtime MaxMinFlag3", MaxMinFlag[3])
        cv2.imshow("Realtime MaxMinFlag4", MaxMinFlag[4])
        cv2.waitKey(1)
    return MaxMinFlag

from math import log, sqrt

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


def GenerateGausPyrPics(I, NumOctaves, scale=1.5, s=3, gaussKernelSize=3, sigma=1.414213):
    pyrPics = []
    NumOctaves += 1
    IntervalNumPerLayers = s + 3
    GaussSigma = genGaussianKernelSigmas(sigma, s)
    for i in range(NumOctaves):
        PicIntervals = []
        PicIntervals.append(I)  # first image in octave already has the correct blur
        for gaussian_kernel in GaussSigma[0:]:
            I = cv2.GaussianBlur(I, (0, 0), sigmaX=gaussian_kernel,
                                 sigmaY=gaussian_kernel)
            PicIntervals.append(I)
        pyrPics.append(PicIntervals)
        octave_base = PicIntervals[-3]
        if i == 0:
            I = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)),
                           interpolation=cv2.INTER_NEAREST)
        else:
            I = cv2.resize(octave_base, (int(octave_base.shape[1] / scale), int(octave_base.shape[0] / scale)),
                           interpolation=cv2.INTER_NEAREST)
    return np.array(pyrPics)


def GenerateDoGImages(pyrPics):
    dog_images = []
    for gaussian_images_in_octave in pyrPics:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(cv2.subtract(second_image,
                                                     first_image))  # ordinary subtraction will not work because the images are unsigned integers
        dog_images.append(dog_images_in_octave)
    return np.array(dog_images)


def genGaussianKernelSigmas(sigma, num_intervals):
    """Generate list of gaussian kernels at which to blur the input image. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper.
    """
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = np.zeros(
        num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels


def FindMaximaMinima(I):
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

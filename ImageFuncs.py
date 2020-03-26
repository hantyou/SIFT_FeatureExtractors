import cv2
import numpy as np


def Direct2D_GaussianBlur(I, SizeOfGaussianKernel=3, sigma=1.414213):
    if not SizeOfGaussianKernel % 2:
        print("Need GaussianKernel Size be Odd")
        quit()
    g1 = cv2.getGaussianKernel(SizeOfGaussianKernel, sigma)
    g2 = cv2.getGaussianKernel(SizeOfGaussianKernel, sigma)
    k = g1 * g2.T
    out = cv2.filter2D(I, -1, kernel=k)
    return out


def Double1D_GaussianBlur(I, SizeOfGaussianKernel=3, sigma=1.414213):
    if not SizeOfGaussianKernel % 2:
        print("Need GaussianKernel Size be Odd")
        quit()
    g1 = cv2.getGaussianKernel(SizeOfGaussianKernel, sigma)
    g2 = cv2.getGaussianKernel(SizeOfGaussianKernel, sigma)
    out = cv2.filter2D(I, -1, kernel=g1)
    out = cv2.filter2D(out, -1, kernel=g2.T)
    return out


def Single1D_GaussianBlur(I, SizeOfGaussianKernel=3, sigma=1.414213, T_orNot=0):
    if not SizeOfGaussianKernel % 2:
        print("Need GaussianKernel Size be Odd")
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
        print("Only parameter larger than 1 is accepted")
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

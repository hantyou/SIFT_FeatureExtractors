import cv2
import matplotlib.pyplot as plt


def Myimshow(str, I):
    cv2.namedWindow(str, cv2.WINDOW_NORMAL)
    cv2.imshow(str, I)


def ShowPicPyr(string, pyrPics):
    len = pyrPics.__len__()
    for i in range(len):
        cv2.namedWindow(string + "Layer" + str(i), cv2.WINDOW_NORMAL)
        cv2.imshow(string + "Layer" + str(i), pyrPics[i])
        print(pyrPics[i].shape)


def ShowPyrPics2(string, pyrPics):
    def ShowPicPyrIner(string, pyrPics):
        len = pyrPics.__len__()
        for i in range(len):
            cv2.namedWindow(string + "Layer" + str(i), cv2.WINDOW_NORMAL)
            cv2.imshow(string + "Layer" + str(i), pyrPics[i])
            print(pyrPics[i].shape)

    len = pyrPics.__len__()
    for i in range(len):
        ShowPicPyrIner(string + ":Octave " + str(i) + ", ", pyrPics[i])


def ShowPyrPics3(string, pyrPics):
    def ShowPicPyrIner(string, pyrPics):
        len = pyrPics.__len__()
        x = 3
        y = int(len / x) + 1
        for i in range(len):
            plt.subplot(y, x, i + 1)
            plt.imshow(pyrPics[i],cmap="gray")
            print(pyrPics[i].shape)
        plt.show()

    len = pyrPics.__len__()
    for i in range(len):
        ShowPicPyrIner(string + ":Octave " + str(i) + ", ", pyrPics[i])

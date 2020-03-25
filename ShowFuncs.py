import cv2

def Myimshow(str, I):
    cv2.namedWindow(str, cv2.WINDOW_NORMAL)
    cv2.imshow(str, I)


def ShowPicPyr(string, pyrPics):
    len = pyrPics.__len__()
    for i in range(len):
        cv2.namedWindow(string + "Pic" + str(i), cv2.WINDOW_NORMAL)
        cv2.imshow(string + "Pic" + str(i), pyrPics[i])
        print(pyrPics[i].shape)


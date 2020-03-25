from cv2 import *

filepath = "28.jpg"
I = imread(filepath)
I = resize(I, (512,512))
g = cvtColor(I,COLOR_BGR2GRAY)
imwrite("gray"+filepath, g)
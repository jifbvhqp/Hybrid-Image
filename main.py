# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from module import Resize,Transform,Fourier,Inv_Fourier,FilterMatrix,idealHighLowPassfilter,Dot,HybridImage,ShowImg

picture_name1 = '4_einstein.bmp'
picture_name2 = '4_marilyn.bmp'

img1 = cv2.imread('data/'+ picture_name1 , cv2.IMREAD_COLOR)
img2 = cv2.imread('data/'+ picture_name2 , cv2.IMREAD_COLOR)
Hybrid_Image = HybridImage(img1,img2,25,10)
ShowImg(Hybrid_Image)
cv2.imwrite('data/'+'output.jpg', Hybrid_Image)
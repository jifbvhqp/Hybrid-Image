# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from module import Resize,Transform,Fourier,Inv_Fourier,FilterMatrix,idealHighLowPassfilter,Dot,HybridImage,ShowImg

picture_name1 = '3_cat.bmp'
picture_name2 = '3_dog.bmp'

img1 = cv2.imread('data/'+ picture_name1 , cv2.IMREAD_COLOR)
img2 = cv2.imread('data/'+ picture_name2 , cv2.IMREAD_COLOR)
Hybrid_Image = HybridImage(img1,img2,25,10)
ShowImg(Hybrid_Image)
#cv2.imwrite('./hw2_data/task1and2_hybrid_pyramid/'+'output.jpg', Hybrid_Image)


	
	
	
	
	
	
	
	
	
	
	
	
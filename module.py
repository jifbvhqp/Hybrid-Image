import cv2
import numpy as np
import matplotlib.pyplot as plt

def Resize(img1,img2):
	M = min(img1.shape[0],img2.shape[0])
	N = min(img1.shape[1],img2.shape[1])
	r = 3
	R_Img1 = np.zeros((M,N,r))
	R_Img2 = np.zeros((M,N,r))
	R_Img1[0:M,0:N,:] = img1[0:M,0:N,:]
	R_Img2[0:M,0:N,:] = img2[0:M,0:N,:]
	return R_Img1,R_Img2

def Transform(img1,img2):
	p,q,r = img1.shape
	for i in range(p):
		for j in range(q):
			if((i+j)%2)==1:
				img1[i,j,:] *= -1
				img2[i,j,:] *= -1
	return img1,img2

def Fourier(img1,img2):
	p,q,r = img1.shape
	FFt_img1 = np.zeros((p,q,r),dtype=np.complex)
	FFt_img2 = np.zeros((p,q,r),dtype=np.complex)
	for i in range(r):
		FFt_img1[:,:,i] = np.fft.fft2(img1[:,:,i])
		FFt_img2[:,:,i] = np.fft.fft2(img2[:,:,i])
	return FFt_img1,FFt_img2
	
def Inv_Fourier(img1,img2):
	p,q,r = img1.shape
	iFFt_img1 = np.zeros((p,q,r),dtype=np.complex)
	iFFt_img2 = np.zeros((p,q,r),dtype=np.complex)
	for i in range(r):
		iFFt_img1[:,:,i] = np.fft.ifft2(img1[:,:,i])
		iFFt_img2[:,:,i] = np.fft.ifft2(img2[:,:,i])
	return iFFt_img1,iFFt_img2

def FilterMatrix(p,q,sigma1,sigma2):
	center_x = int(p/2)
	center_y = int(q/2)
	H1 = np.zeros((p,q))
	H2 = np.zeros((p,q))
	for i in range(p):
		for j in range(q):
			x = (i-center_x)**2
			y = (j-center_y)**2
			H1[i,j] = 1-np.exp(-(x+y)/(2*sigma1*sigma1))
			H2[i,j] = np.exp(-(x+y)/(2*sigma2*sigma2))
	return H1,H2

def idealHighLowPassfilter(p,q,cutoff):
	center_x = int(p/2)
	center_y = int(q/2)
	H1 = np.zeros((p,q))
	H2 = np.zeros((p,q))
	for i in range(p):
		for j in range(q):
			x = (i-center_x)**2
			y = (j-center_y)**2
			if (x + y)**1/2 <= cutoff:
				H1[i,j] = 0
			else:
				H1[i,j] = 1
				
				
			if (x + y)**1/2 <= cutoff:
				H2[i,j] = 1
			else:
				H2[i,j] = 0
			
	return H1,H2
	
def Dot(img1,img2,H1,H2):
	p,q,r = img1.shape
	Dot_img1 = np.zeros((p,q,r),dtype=np.complex)
	Dot_img2 = np.zeros((p,q,r),dtype=np.complex)
	for i in range(r):
		Dot_img1[:,:,i] = img1[:,:,i]*H1
		Dot_img2[:,:,i] = img2[:,:,i]*H2
	return Dot_img1,Dot_img2

def HybridImage(img1,img2,sigma1,sigma2):
	m,n = img1.shape[0:2]
	img1,img2 = Resize(img1,img2) #Make img1 and img2 to be same size
	
	img1,img2 = Transform(img1,img2) #(-1)^(x+y)
	img1,img2 = Fourier(img1,img2)
	
	H1,H2 = FilterMatrix(img1.shape[0],img1.shape[1],sigma1,sigma2)
	#H1,H2 = idealHighLowPassfilter(img1.shape[0],img1.shape[1],80)
	
	img1,img2 = Dot(img1,img2,H1,H2)
	img1,img2 = Inv_Fourier(img1,img2)
	
	img1 = np.real(img1)
	img2 = np.real(img2)
	img1,img2 = Transform(img1,img2) #(-1)^(x+y)

	return img1 + img2

def ShowImg(img):
	img = img[:,:,::-1] #BGR to RGB
	plt.figure("Image")
	plt.imshow(img/255)
	plt.axis('off')
	plt.title('image')
	plt.show()
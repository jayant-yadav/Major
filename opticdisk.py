import cv2
import numpy as np


res = cv2.imread('17_right.jpeg',1)
img = cv2.resize(res,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
#cv2.imshow('image',img)
# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#green = img[0,:,0]
#cv2.imshow('gray',green)
h,s,v=cv2.split(hsv)
median = cv2.medianBlur(v,5)
equ = cv2.equalizeHist(median)
# global thresholding
cv2.imshow('equ',equ)
ret1,th1 = cv2.threshold(equ,250,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(equ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
#blur = cv2.GaussianBlur(g,(5,5),0)
#ret3,th3 = cv2.threshold(equ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
f=equ-th1
cv2.imshow('image',th1)
cv2.waitKey(0)
cv2.destroyAllWindows()





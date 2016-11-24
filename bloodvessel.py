import cv2
import numpy as np


img = cv2.imread('image002.jpg',1)
cv2.imshow('initial',img)

#cv2.imshow('image',img)
# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#green = img[0,:,0]
#cv2.imshow('gray',green)
h,s,v=cv2.split(hsv)
median = cv2.medianBlur(v,5)
equ = cv2.equalizeHist(median)
kernel1 = np.ones((5,5),np.uint8)
kernel2 = np.ones((3,3),np.uint8)
b,g,r=cv2.split(img)
cv2.imshow('green',g)
closing1 = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel1)
closing2 = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel2)

# plot all the images and their histograms
gradient_image=closing1-closing2
cv2.imshow('gradient_image',gradient_image)
blur = cv2.GaussianBlur(gradient_image,(5,5),0)
ret3,gradient_image_th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('image',gradient_image_th)
#res = cv2.bitwise_and(gradient_image_th,gradient_image_th, mask= gradient_image)
#cv2.imshow('final',res);
cv2.waitKey(0)
cv2.destroyAllWindows()





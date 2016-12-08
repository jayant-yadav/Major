import cv2
import numpy as np
import Image

#img = Image.open('image032.png')
#img.save('image032.jpg')
img = cv2.imread('image032.jpg',1)
cv2.imshow('initial',img)
bv002=cv2.imread('bv018.jpg',1)
optic002=cv2.imread('opticdisk018.jpg',1)
#optic002=cv2.imread('image032.jpg',1)
cv2.imshow('bv',bv002)
#cv2.imshow('image',img)
# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#print(np.array(img).shape)
#print(np.array(bv002).shape)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#green = img[0,:,0]
#cv2.imshow('gray',green)
h,s,v=cv2.split(hsv)
median = cv2.medianBlur(v,5)
equ = cv2.equalizeHist(median)
kernel1 = np.ones((7,7),np.uint8)
kernel2 = np.ones((5,5),np.uint8)
kernel3 = np.ones((4,4),np.uint8)
b,g,r=cv2.split(img)
cv2.imshow('green',g)
b1,g1,r1=cv2.split(bv002)
b2,g2,r2=cv2.split(optic002)
cv2.imshow('green1',g1)
dilation1 = cv2.dilate(g,kernel1)
dilation2 = cv2.dilate(g,kernel2)


# plot all the images and their histograms
gradient_image=np.array(dilation1)-np.array(dilation2)
cv2.imshow('gradient_image',gradient_image)
#blur = cv2.GaussianBlur(gradient_image,(5,5),0)
#cv2.imshow('blur',blur)
#ret3,gradient_image_th = cv2.threshold(gradient_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#cv2.imshow('image',gradient_image_th)
#gradient_image_th_eroded=cv2.erode(gradient_image_th,kernel3,iterations = 1)
#res = cv2.bitwise_and(gradient_image_th,gradient_image_th, mask= gradient_image)
#cv2.imshow('final',res);
#gradient_image_th_eroded_erode=cv2.erode(gradient_image_th_eroded,kernel3,iterations = 1)
#cv2.imshow('gradient_image_th_eroded',gradient_image_th_eroded)
#cv2.imshow('gradient_image_th_eroded_erode',gradient_image_th_eroded_erode)
#opening = cv2.morphologyEx(gradient_image_th_eroded, cv2.MORPH_OPEN, kernel1)
#closing = cv2.morphologyEx(gradient_image_th_eroded, cv2.MORPH_CLOSE, kernel1)
#cv2.imshow('Opening',opening)
#cv2.imshow('Closing',closing)

#print(np.array(gradient_image_th).shape)
#print(np.array(bv002).shape)
gradient_image_th_wbv=cv2.subtract(gradient_image,g1)
cv2.imshow('gradient_image_th_wbv',gradient_image_th_wbv)
gradient_image_th_wbvod=cv2.subtract(gradient_image_th_wbv,g2)
#im = cv2.imfill(gradient_image_th,'holes');
cv2.imshow('gradient_image_th_wbvod',gradient_image_th_wbvod)

cv2.waitKey(0)
cv2.destroyAllWindows()





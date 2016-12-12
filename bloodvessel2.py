import cv2
import numpy as np
import Image

#img = Image.open('image032.png')
#img.save('image032.jpg')
res = cv2.imread('image002.jpg',1)
#img = cv2.resize(res,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
img = cv2.resize(res,(640,491))
cv2.imshow('initial',img)

#cv2.imshow('image',img)
# Convert BGR to HSV
#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#green = img[0,:,0]
#cv2.imshow('gray',green)
#h,s,v=cv2.split(hsv)
#median = cv2.medianBlur(v,5)
#equ = cv2.equalizeHist(median)
kernel1 = np.ones((7,7),np.uint8)
kernel2 = np.ones((3,3),np.uint8)
kernel3 = np.ones((4,4),np.uint8)
b,g,r=cv2.split(img)

cv2.imshow('green',g)
closing1 = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel1)
closing2 = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel2)
cv2.imshow('closing1',closing1)
cv2.imshow('closing2',closing2)
###############33
greensubclosing=cv2.subtract(closing1,g)
cv2.imshow('green-closing1',greensubclosing)
###############3
# plot all the images and their histograms
#gradient_image=np.array(closing1)-np.array(closing2)
gradient_image=greensubclosing
cv2.imshow('gradient_image',gradient_image)

#gradient_image_th = cv2.adaptiveThreshold(gradient_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 0)
ret,gradient_image_th = cv2.threshold(gradient_image,2,255,cv2.THRESH_BINARY)
#blur = cv2.GaussianBlur(gradient_image,(5,5),0)
#ret3,gradient_image_th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('image',gradient_image_th)
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

cv2.imshow('final',cv2.subtract(g,gradient_image_th))
#cv2.imwrite('image002_bv.jpg',cv2.subtract(g,gradient_image_th))
cv2.imwrite('bv018.jpg',gradient_image_th)
#detecting exudates


cv2.waitKey(0)
cv2.destroyAllWindows()





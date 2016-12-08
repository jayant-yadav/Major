import cv2
import numpy as np
import Image

#img = Image.open('image032.png')
#img.save('image032.jpg')
res = cv2.imread('image032.jpg',1)
#img = cv2.resize(res,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
img = cv2.resize(res,(640,491))
cv2.imshow('initial',img)

#cv2.imshow('image',img)
# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#green = img[0,:,0]
#cv2.imshow('gray',green)
#h,s,v=cv2.split(hsv)
#median = cv2.medianBlur(v,5)
#equ = cv2.equalizeHist(median)
kernel1 = np.ones((5,5),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
kernel3 = np.ones((4,4),np.uint8)
#kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))	
#kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))	
b,g,r=cv2.split(img)
cv2.imshow('green',g)
median = cv2.medianBlur(g,3)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(median)
cv2.imshow('clahe',cl1)
closing1 = cv2.morphologyEx(cl1, cv2.MORPH_CLOSE, kernel1)
closing2 = cv2.morphologyEx(cl1, cv2.MORPH_CLOSE, kernel2)

# plot all the images and their histograms
gradient_image=np.array(closing1)-np.array(closing2)
cv2.imshow('gradient_image',gradient_image)
blur = cv2.GaussianBlur(gradient_image,(5,5),0)
ret3,gradient_image_th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('image',gradient_image_th)
gradient_image_th_eroded=cv2.erode(gradient_image_th,kernel3,iterations = 1)
#res = cv2.bitwise_and(gradient_image_th,gradient_image_th, mask= gradient_image)
#cv2.imshow('final',res);
#gradient_image_th_eroded_erode=cv2.erode(gradient_image_th_eroded,kernel3,iterations = 1)
cv2.imshow('gradient_image_th_eroded',gradient_image_th_eroded)
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





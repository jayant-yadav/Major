import cv2
import numpy as np


res = cv2.imread('16_right.jpeg',1)
img = cv2.resize(res,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
#cv2.imshow('image',img)
# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('grayscale',gray)

#kernel = np.ones((26,26),np.uint8) #taking circle of radius 13
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(26,26)) #taking ellipse of size 26
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))	
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))	
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel) #dilation followed by erosion
cv2.imshow('closing',closing)
#Mat double_filtering
#closing.convertTo( double_filtering, CV_64FC3, 1.0/255.0 );
double_precision = cv2.normalize(closing.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
cv2.imshow('double_precision',double_precision)
blur = cv2.GaussianBlur(closing,(5,5),0)
cv2.imshow('blur',blur)
#thresh= cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #to filter out exudates
ret1,thresh = cv2.threshold(blur,210,255,cv2.THRESH_BINARY)
cv2.imshow('threshold',thresh)
dilated_disk = cv2.dilate(thresh,kernel,iterations = 1)
#cv2.imshow('dialted_disk',dilated_disk)

no_disk=cv2.subtract(gray,dilated_disk)
cv2.imshow('disk',no_disk)

closing_vessel1 = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
closing_vessel2 = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2)
closing_vessel=cv2.subtract(closing_vessel1,closing_vessel2)
cv2.imshow('closing_vessel', closing_vessel)
ret2,blood_vessel_thresh= cv2.threshold(closing_vessel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#vessel_erosion = cv2.erode(blood_vessel_thresh,kernel2,iterations = 1)
#dilation = cv2.dilate(blood_vessel_thresh,kernel2,iterations = 1)
vessel_erosion = cv2.medianBlur(blood_vessel_thresh,5)
cv2.imshow('vessels',blood_vessel_thresh)
cv2.imshow('vessels erosion',vessel_erosion)

ret3,eye_edge_thresh= cv2.threshold(closing,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
eye_edge=cv2.Canny(eye_edge_thresh, 0,0, 0 )
cv2.imshow('eye edge', eye_edge)

blood_vessel_edge=cv2.Canny(blood_vessel_thresh, 0,0, 0 )
cv2.imshow('vessel edge detection', blood_vessel_edge)






cv2.waitKey(0)
cv2.destroyAllWindows()


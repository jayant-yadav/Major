import numpy as np
import cv2

img1=cv2.imread('16_right.jpeg')
orig=img1.copy()
img = cv2.resize(img1,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(51,51),0) #size 3,3 
#radius of gaussian blur must be odd
#area of the image with largest intensity value
(minVal,maxVal,minLoc,maxLoc)=cv2.minMaxLoc(gray)
cv2.circle(img,maxLoc,50,(255,0,0),2)

cv2.imshow("Robust",img)




cv2.waitKey(0)
cv2.destroyAllWindows()


#import the necessary packages
import numpy as np
#import argparse
import cv2
import Image

img = Image.open('image013.png')
img.save('image013.jpg') 
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", help = "path to the image file")
#ap.add_argument("-r", "--radius", type = int,
#	help = "radius of Gaussian blur; must be odd")
#args = vars(ap.parse_args())
 
# load the image and convert it to grayscale

image = cv2.imread('image013.jpg',1)
height,width,depth = image.shape
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
b,g,r=cv2.split(image)
# the area of the image with the largest intensity value
#(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
#cv2.circle(image, maxLoc, 5, (255, 0, 0), 2)
 
# display the results of the naive attempt
#cv2.imshow("Naive", image)

# apply a Gaussian blur to the image then find the brightest
# region
gray = cv2.GaussianBlur(g, (55, 55), 0)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(g)
image = g.copy()
cv2.circle(image, maxLoc,55 , 1, 2)


circle_img = np.zeros((height,width), np.uint8)
cv2.circle(circle_img,maxLoc,55,1,thickness=-1)

masked_data = cv2.bitwise_and(image, image, mask=circle_img)
#masked_data = cv2.bitwise_or(image, image, mask=circle_img)
ret,thresh1 = cv2.threshold(masked_data,0,255,cv2.THRESH_BINARY)
cv2.imshow("masked", masked_data)
cv2.imshow("thresh1", thresh1)
cv2.imwrite('opticdisk018.jpg',thresh1) 
# display the results of our newly improved method
cv2.imshow("Robust", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

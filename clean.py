import cv2
import numpy as np
  
#image = cv2.imread('C:\\Users\\Ashwath\\Desktop\\Capstone\\Unet lung segmentation\\UNET\\results\\MontgomerySet\\CXR_png\\MCUCXR_0017_0.png')
 
image = cv2.imread(`image Path`)

# Gaussian Blur
Gaussian = cv2.GaussianBlur(image, (7, 7), 0)
cv2.imshow('Gaussian Blurring', Gaussian)
cv2.waitKey(0)
  
# Median Blur
median = cv2.medianBlur(image, 5)
cv2.imshow('Median Blurring', median)
cv2.waitKey(0)
  
  
# Bilateral Blur
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('Bilateral Blurring', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()
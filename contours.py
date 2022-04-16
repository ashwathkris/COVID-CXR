import cv2
import numpy as np
from PIL import Image
for i in range(2,14):
    image1 = cv2.imread(str(i) + '/pred.png')   #img1
    image2 = cv2.imread(str(i) + '/man.png')    #img2
    xray = cv2.imread(str(i)   + '/xray.png')

    xray = cv2.resize(xray, (512, 512))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)    #img1
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)    #img2
      
    # Find Canny edges
    edged1 = cv2.Canny(gray1, 30, 200)  #img1
    edged2 = cv2.Canny(gray2, 30, 200)  #img2

    img, contours1, hierarchy = cv2.findContours(edged1,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   #img1
    img, contours2, hierarchy = cv2.findContours(edged2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   #img2

    cv2.drawContours(xray, contours1, -1, (0, 255, 0), 1)   #img1
    cv2.drawContours(xray, contours2, -1, (0, 0, 255), 1)   #img2
      
    cv2.imshow('Contours', xray)
    cv2.waitKey(0)
cv2.destroyAllWindows()
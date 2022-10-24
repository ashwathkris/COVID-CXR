import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pywt
import pywt.data

imgs = ['c1.png','c4.png','nc.png', 'c3.png']
for img in imgs:
    #image = cv2.imread("c3.png")
    image1 = cv2.imread(img)
    image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    cv2.imshow(img, image)

    #image = cv2.equalizeHist(image)
    (LL, (LH, HL, HH)) = pywt.dwt2(image, 'bior1.3')
    LL_comp = Image.fromarray(LL)
    LL_comp = LL_comp.convert('RGB')

    ##CONVERT TO NUMPY ARRAY
    LL_comp = np.array(LL_comp)

    #RESIZE
    img1 = cv2.resize(LL_comp, (256, 256))
    cv2.imshow('compression', img1)

    img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img)
    cv2.imshow('hist_eq', equ)

    ##EROSION
    #kernel = np.ones((3,3), np.uint8)
    #img_erosion = cv2.erode(equ, kernel, iterations=2)
    #cv2.imshow('Erosion', img_erosion)


    #THRESHOLDING
    #(thresh, im_bw) = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    #(thresh, im_bw) = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_bw = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
    cv2.imshow('THRESHOLD', im_bw)
    


#########
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(im_bw, None, None, None, 8, cv2.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 200:   #keep
            result[labels == i + 1] = 255

    #cv2.imshow("cleaned", result)

    ###DRAWING
    edges = cv2.Canny(result,100,200) 
    #cv2.imshow('canny', edges)

    #CONTOURS
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image1, contours, -1, (0, 255, 0), 1)
    cv2.imshow('Infection', image1)

#########

    res =  equ & im_bw
    cv2.imshow('res', res)


    cv2.waitKey(0)

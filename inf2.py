import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pywt
import pywt.data
#LOAD IMAGE


imgs = ['Covid (1).png','Covid (2).png','Covid (3).png', 'Covid (18).png','Non-Covid (4).png','Non-Covid (36).png',]
i=0
for j in imgs:
    img_name = j
    input1 = cv.imread(img_name)
    input = cv.resize(input1, (256, 256))
    cv.imshow(img_name, input)
    img = cv.GaussianBlur(input,(3,3), 0)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #cv.imshow('img', img)

    #HISTOGRAM EQUALIZATION
    equ = cv.equalizeHist(img)
    #cv.imshow('hist_eq', equ)

    #BLURRING
    img = cv.GaussianBlur(equ,(3,3), 0)
    #cv.imshow('Blurring', img)

    #THRESHOLDING
    (thresh, im_bw) = cv.threshold(img, 80, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    #cv.imshow('THRESHOLD', im_bw)

    #CANNY
    edges = cv.Canny(im_bw,100,200) 
    #cv.imshow('canny', edges)

    #CONTOURS
    contours, hierarchy = cv.findContours(im_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    #cv.drawContours(input, contours, -1, (0, 255, 0), 1)
    #cv.imshow('contours', input)


    ############# END OF IMAGE PROCESSING ######################

    def get_contour_areas(contours):
        all_areas= []
        for cnt in contours:
            area= cv.contourArea(cnt)
            all_areas.append(area)
        return all_areas


    sorted_contours= sorted(contours, key=cv.contourArea, reverse= True)
    if(len(sorted_contours)>1):
        big1= sorted_contours[0]
        big2= sorted_contours[1]
        cv.drawContours(input, big1, -1, (0,255,0),2)
        cv.drawContours(input, big2, -1, (0,255,0),2)

    else:
        big1= sorted_contours[0]
        cv.drawContours(input, big1, -1, (0,255,0),2)
    


    res = np.zeros((256,256), dtype = 'uint8')
    if(len(sorted_contours)>1):
        cv.fillPoly(res, [big1] , color=(255,255,255))
        cv.fillPoly(res, [big2] , color=(255,255,255))
    else:
        cv.fillPoly(res, [big1] , color=(255,255,255))

    #cv.imshow('Lung Masks', res)

    input = cv.cvtColor(input,cv.COLOR_BGR2GRAY)
    #cv.imshow('Lungs', input)

    final = cv.bitwise_and(input, input, mask=res)
    cv.imshow('Overlap', final)




#######END OF SEGMENTATION###############


#######END OF SEGMENTATION###############
    def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
        
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            
            buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
        
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            
            buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    image = final

    improved_contrast = apply_brightness_contrast(image, 0, 64)
    cv.imshow('improved contrast', improved_contrast)
    #image = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
   #cv.imshow(img, image)
    #image = cv.equalizeHist(image)
    (LL, (LH, HL, HH)) = pywt.dwt2(image, 'bior1.3')
    LL_comp = Image.fromarray(LL)
    LL_comp = LL_comp.convert('RGB')

    ##CONVERT TO NUMPY ARRAY
    LL_comp = np.array(LL_comp)

    #RESIZE
    img1 = cv.resize(LL_comp, (256, 256))
    cv.imshow('compression', img1)
    img = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    equ = cv.equalizeHist(img)
    cv.imshow('hist_eq', equ)
    # filterSize = (3, 3)
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, filterSize)
    # blackhat_img = cv.morphologyEx(equ, cv.MORPH_BLACKHAT, kernel)
    # tophat_img = cv.morphologyEx(equ, cv.MORPH_TOPHAT, kernel)
    # cv.imshow('black hat', blackhat_img)
    # cv.imshow('top hat', tophat_img) 
    histogram = cv.calcHist([equ], [0], res, [256], [0, 256])
    

    prev_val = histogram[255][0]
    thres = None
    for i in range(histogram.shape[0] - 1, 0, -1):
        # print(histogram[i][0])
        val = histogram[i][0]
        if val > prev_val:
            thres = i + 1
            break;
        prev_val = val
    print(thres)
    # plt.plot(histogram, color='k')
    # plt.show()


    #THRESHOLDING
    (thresh, im_bw) = cv.threshold(equ, thres - 15, 255, cv.THRESH_BINARY)
    # (thresh, im_bw) = cv.threshold(equ, thres, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # im_bw = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,5,2)
    cv.imshow('THRESHOLD', im_bw)

    res =  equ & im_bw
    cv.imshow('res', res)

    # histogram = cv.calcHist([res], [0], mask, [256], [0, 256])
    # plt.plot(histogram, color='k')
    # plt.show()

    for i in range(256):
        for j in range(256):
            if(res[i][j] <= 20):
                res[i][j] = 0

    cv.imshow('Remove grey areas', res)

    im_bw = res


#########
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(improved_contrast, None, None, None, 8, cv.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 20:   #keep
            result[labels == i + 1] = 255

    cv.imshow("Cleaned", result)

    ###DRAWING
    edges = cv.Canny(result,100,200) 
    #cv2.imshow('canny', edges)

    input_image = cv.imread(img_name)
    input_image = cv.resize(input_image, (256, 256))

    #CONTOURS
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(input_image, contours, -1, (0, 255, 0), 1)
    cv.imshow('Infection', input_image)

#########

    cv.waitKey(0)

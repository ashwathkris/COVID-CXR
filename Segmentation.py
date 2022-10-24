import cv2 as cv
import numpy as np
#LOAD IMAGE

covid = ['Covid (1).png', 'Covid (2).png', 'Covid (3).png', 'Covid (4).png','Covid (5).png','Covid (6).png', 'Covid (7).png']
noncovid = ['Non-Covid (4).png', 'Non-Covid (36).png']

test = ['Covid (18).png']
i=0
for j in test:
    img_name = j
    input1 = cv.imread(img_name)
    input = cv.resize(input1, (256, 256))
    cv.imshow('Input Image', input)
    img = cv.GaussianBlur(input,(3,3), 0)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cv.imshow('img', img)

    #HISTOGRAM EQUALIZATION
    equ = cv.equalizeHist(img)
    cv.imshow('hist_eq', equ)

    #BLURRING
    img = cv.GaussianBlur(equ,(3,3), 0)
    cv.imshow('Blurring', img)

    #THRESHOLDING
    (thresh, im_bw) = cv.threshold(img, 80, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow('THRESHOLD', im_bw)

    #CANNY
    edges = cv.Canny(im_bw,100,200) 
    cv.imshow('canny', edges)

    #CONTOURS
    contours, hierarchy = cv.findContours(im_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(input, contours, -1, (0, 255, 0), 1)
    cv.imshow('contours', input)

    cv.waitKey(0)

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
    
    cv.waitKey(0)
    cv.imshow('Lungs', input)
    cv.waitKey(0)


    res = np.zeros((256,256), dtype = 'uint8')
    if(len(sorted_contours)>1):
        cv.fillPoly(res, [big1] , color=(255,255,255))
        cv.fillPoly(res, [big2] , color=(255,255,255))
    else:
        cv.fillPoly(res, [big1] , color=(255,255,255))

    cv.imshow('Lung Masks', res)

    input = cv.cvtColor(input,cv.COLOR_BGR2GRAY)
    cv.imshow('Lungs', input)

    final = cv.bitwise_and(input, input, mask=res)
    cv.imshow('Overlap', final)

    cv.imwrite(str(i)+j, final)
    i+=1
#-------------------------------END OF SEGMENTATION-----------------------------------------

    #(thresh, im_bw) = cv.threshold(final, 80, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    #cv.imshow('THRESHOLD', im_bw)

    #CANNY
    #edges = cv.Canny(final,100,200) 
    #cv.imshow('canny', edges)




    cv.waitKey(0)



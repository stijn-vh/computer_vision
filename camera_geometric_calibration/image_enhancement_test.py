# Source:
# https://stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencv
import glob

import cv2 as cv

manual_images = glob.glob('images/automatic/*.jpg')
for img_path in manual_images:
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    smoothed = cv.GaussianBlur(img, (0, 0), 3)
    improved = cv.addWeighted(img, 2, smoothed, -1, 0) #params alpha/beta/gamma

    ret1, ogcorners = cv.findChessboardCorners(img, (9, 6), None)
    #print("original returns ", ret1)
    ret2, smcorners = cv.findChessboardCorners(smoothed, (9, 6), None)
    #print("smoothed returns ", ret2)
    ret3, imcorners = cv.findChessboardCorners(improved, (9, 6), None)
    #print("improved returns ", ret3, "\n\n")

    if ret1 != ret3:
        print("original returns ", ret1)
        print("improved returns ", ret3, "\n\n")
        cv.imshow('original', img)
        cv.imshow('smoothed', smoothed)
        cv.imshow('improved', improved)

cv.waitKey(100000)
cv.destroyAllWindows()

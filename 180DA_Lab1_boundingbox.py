# REFERENCES:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
# https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
# https://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # convert bgr to rbg and hsv
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of yellow color in RGB
    lower_yellow_rgb = np.array([230,160,67])
    upper_yellow_rgb = np.array([244,255,20])

    # define range of yellow color in HSV
    lower_yellow_hsv = np.array([34,71,90])
    upper_yellow_hsv = np.array([63,87,100])

    # Threshold the RGB image to get only yellow colors
    mask_rgb = cv2.inRange(rgb, lower_yellow_rgb, upper_yellow_rgb)
    # Threshold the HSV image to get only yellow colors
    mask_hsv = cv2.inRange(hsv, lower_yellow_hsv, upper_yellow_hsv)

    contours_rgb, hierarchy_rgb = cv2.findContours(mask_rgb,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours_hsv, hierarchy_hsv = cv2.findContours(mask_hsv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
    for c in contours_rgb:
        # get the min area rect
        #rect = cv2.minAreaRect(c)

        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    for c in contours_hsv:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a blue rectangle to visualize the bounding rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)


    cv2.imshow('frame',frame)
    cv2.imshow('RGB mask',mask_rgb)
    cv2.imshow('HSV mask',mask_hsv)
    #cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


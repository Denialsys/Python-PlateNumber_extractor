'''
    Author: Jenver I.

    For extraction of  license plates from vehile using haar cascades
    Cascades downloadable from this link: https://github.com/opencv/opencv/tree/master/data/haarcascades
    Tutorial here: https://www.youtube.com/watch?v=yMQvcWBx1fE

    To do:
        - Open a dashcam video
        - Feed each frame to the license extractor
        - Extract and record the data found then take screenshot
'''


import cv2
import numpy as np

cascade = cv2.CascadeClassifier("cascades/haarcascade_russian_plate_number.xml")
read = ""

def extract_plate(img_name):
    global read
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in nplate:
        a, b = ( int(0.02*img.shape[0]), int(0.025*img.shape[1]))
        plate = img[y+a:y+h-a, x+b:x+w-b, ;]
        kernel = np.ones((1,1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        (thresh, plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)


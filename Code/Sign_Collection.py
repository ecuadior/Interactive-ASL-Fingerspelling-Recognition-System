#Collect ASL Hand signs
#Images that used to train classfier will be a cropped image of the hand performing the image

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0) #camera to record dat
detection = HandDetector(maxHands=1)# 1 hand being tracked due to asl finger spelling is mostly one hand

folder = "Data/Z" #access to folder in which crop images of asl fingerspelling where will be stored
counter = 0 #Counts how many image will be saved. Approx 1000 or so images will be saved. Thereofe in total 26,000 images will be saved

offset_cp = 35 # for cropped image
imgSize = 300 #used to create white backgrpound for croped image of data

while True:
    success, img = cap.read()
    hands, img = detection.findHands(img) #show hands being detected in window being open

    if hands: #if hand is detected, cropping the image, also if hands to close to the camer it stop running
        hand = hands[0] # 1 hands, [n-1] is for amount of hand being read
        x,y,w,h = hand['bbox'] # box create for the hand

        imgWhite = np.ones((imgSize, imgSize,3),np.uint8)*255 #white bg for image of data
        imgCrop = img[y-offset_cp:y+h+offset_cp,x-offset_cp:x+w+offset_cp] #gives bounding box require
        imgCropShape = imgCrop.shape
        aspectRatio = h/w #aspect ratio of the hand

        #adding cropped image of hand to imgWhite
        if aspectRatio > 1: #if hand is more vertical than horizontal
            k = imgSize/h 
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize #overlay cropimsge on wb
        else:# if hand is more horizontal then  vertical
            k = imgSize/w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize #overlay cropimsge on


        cv2.imshow("Cropped Image", imgCrop) #image cropped(not being used for data)
        cv2.imshow("Data Collecting", imgWhite) #white bg gui

    cv2.imshow("test",img)
    ImageKey =cv2.waitKey(1)
    

    if ImageKey == ord('s'): # when s is press n amount of frames will be saved to folder
        counter+=1 
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite) #cropped image in white bg being saved to folder
        print(counter) 


    



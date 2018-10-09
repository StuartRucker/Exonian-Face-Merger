import numpy as np
import cv2 as cv
import os

count = 0
final_images = []

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')


for file in os.listdir("girls"):
    count = count + 1

    try:

        img = cv.imread('girls/' + file)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # for (x,y,w,h) in faces:
        #     cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #     roi_gray = gray[y:y+h, x:x+w]
        #     roi_color = img[y:y+h, x:x+w]
        #     eyes = eye_cascade.detectMultiScale(roi_gray)
        #     for (ex,ey,ew,eh) in eyes:
        #         cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        # cv.imshow('img',img)
        # cv.waitKey(0)
        # #

        blank = np.zeros([600,600],dtype=np.uint8)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) == 2 and len(faces) == 1:
                ex1,ey1,ew1,eh1 = eyes[0]
                ex2,ey2,ew2,eh2 = eyes[1]


                y_offset = 230-ey1-y
                x_offset = 230-ex1-x
                if(ex2 < ex1):
                    y_offset = 230-ey2-y
                    x_offset = 230-ex2-x

                distortionfactor = 50./(abs(ex2-ex1))
                # print(distortionfactor)
                resized_image = gray #cv.resize(gray, None, fx = distortionfactor, fy = distortionfactor, interpolation = cv.INTER_CUBIC)

                blank[y_offset:y_offset+resized_image.shape[0], x_offset:x_offset+resized_image.shape[1]] = resized_image

                final_images.append(blank)
    except:
        pass

blank_final = np.mean(final_images, axis=0)/255.

cv.imshow('img',blank_final)
cv.waitKey(0)
cv.destroyAllWindows()

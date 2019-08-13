from imutils.video import VideoStream
from imutils import face_utils
import imutils
from scipy.spatial import distance
import playsound
from threading import Thread
import argparse
import numpy
import time
import dlib
import cv2


#a function calculating the ratio of vertical distance over horizontal distance of the passed in eye
def eye_ratio(eye):
    vert1 = distance.euclidean(eye[1],eye[5])
    vert2 = distance.euclidean(eye[2],eye[4])
    horiz = distance.euclidean(eye[0],eye[3])

    ratio = (vert1 + vert2)/(2.0*horiz)
    return ratio

#parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p','--shape-predictor',required=True, help='path to pre-trained facial landmark detector')
parser.add_argument('-a','--alarm',type=str,default='',help='path to alarm mp3')
args = vars(parser.parse_args())

#CONSTANTS (adjust the values as you see fit)
EYE_RATIO_THRESHOLD = 0.23 #threshold for eye ratio to reach below
COUNT_THRESHOLD = 15 #threshold for frame count to get past

#count that keeps track of consecutive frames with shut eyes
count = 0



#Initialize your face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

#store index ranges of left eye and right eye
(startLeft,endLeft) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(startRight,endRight) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']



#initialize the videostream and start
vs = VideoStream(src=0).start()
time.sleep(2.0)

#keep processing frames indefinitely 
while True:
    #the current frame is processed as a read image, resized, then grayscaled for better pixel intensity evaluation through the predictor
    frame = vs.read()
    frame = imutils.resize(frame,width = 600)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detects and stores the face coordinates in rect
    rect = detector(gray)
    if len(rect) > 0:
        #extracts the facial landmarks from rect
        shape = predictor(gray,rect[0])
        shape = face_utils.shape_to_np(shape)


        #get the left and right eye landmark points
        leftEye = shape[startLeft:endLeft]
        rightEye = shape[startRight:endRight]

        #calculate the ratios of the left and right eye, and get the average
        leftRatio = eye_ratio(leftEye)
        rightRatio = eye_ratio(rightEye)
        average_ratio = (leftRatio + rightRatio)/(2.0)

        #When the eyes close down to go past the threshold, start counting
        if average_ratio < EYE_RATIO_THRESHOLD:
            count += 1
            #once the count goes past its threshold, sound the alarm indicating you're dozing off
            if count >= COUNT_THRESHOLD:
                t = Thread(target=playsound.playsound, args=(args['alarm'],)) 
                t.deamon = True
                t.start()  
                cv2.putText(frame,'WARNING: YOU ARE DOZING!',(150,150),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        #When the eyes start to or continue to be open, then reset count to 0
        else:
            count = 0

        #code here is for the added effect of showing the eye landmarks detected from predictor
        hullLeft = cv2.convexHull(leftEye)
        hullRight = cv2.convexHull(rightEye)
        cv2.drawContours(frame,[hullLeft],-1,(0,255,0),1)
        cv2.drawContours(frame,[hullRight],-1,(0,255,0),1)
        #shows your current eye ratio, and gives you a good idea of how open your eyes appear to the predictor
        cv2.putText(frame,'EAR: {}'.format(average_ratio),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    #After image processing the frame, show the frame onto a window, which is refreshed every 1 millisecond (you can change the timing in parameter of cv2.waitKey())
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(1) & 0xFF
    
    #When you press q, the program breaks out of the image processing while loop then destroys all generated windows and stops the videostream.
    if key == ord('q'):
        break
cv2.destroyAllWindows()
vs.stop()




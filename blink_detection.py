from imutils.video import VideoStream
from imutils import face_utils
import imutils
import argparse
from scipy.spatial import distance
import time
import numpy
import dlib
import cv2


#calculates the ratio of horizontal distance over vertical distance of eye
def eye_ratio(eye):
    vert1 = distance.euclidean(eye[1],eye[5])
    vert2 = distance.euclidean(eye[2],eye[4])
    horiz = distance.euclidean(eye[0],eye[3])

    ratio = (vert1 + vert2)/(2.0*horiz)
    return ratio


#parse command-line argument containing path to shape-predictor
parser = argparse.ArgumentParser()
parser.add_argument('-p','--shape-predictor',required=True,help='path to facial landmark detector')
args = vars(parser.parse_args())


#constants and global variables
EYE_RATIO_THRESHOLD = 0.25
COUNT_THRESHOLD = 3

count = 0
total = 0

#initializing face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

#set up the indexes of where the eye landmarks are
(startLeft, endLeft) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(startRight, endRight) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#use videostream to get our frames
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    #read the current frame, then pre-process it to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #rects is the collection of detected faces in the current frame
    rects = detector(gray)
    for rect in rects:
        #get the landmark points of the current face detected
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)
        
        #extract the left and right eye landmark points
        left = shape[startLeft:endLeft]
        right = shape[startRight:endRight]

        #calculate the ratios of the left and right eye landmarks passed in then get the average
        leftRatio = eye_ratio(left)
        rightRatio = eye_ratio(right) 
        average_ratio = (leftRatio + rightRatio)/(2.0)

        #increment count for each frame if the ratio is less than the threshold (eyes are closed)
        if average_ratio < EYE_RATIO_THRESHOLD:
            count += 1
        #otherwise (when eyes are open), check if the count has reached the threshold, and increment total. Reset count to 0 regardless.
        else:
            if count >= COUNT_THRESHOLD:
                total += 1
            count = 0 

        cv2.putText(frame, 'Blink#{}'.format(total),(10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.putText(frame, 'EAR: {}'.format(average_ratio),(300,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

        hullLeft = cv2.convexHull(left)
        hullRight = cv2.convexHull(right)

        cv2.drawContours(frame, [hullLeft], -1, (0,255,0),1)
        cv2.drawContours(frame, [hullRight], -1, (0,255,0),1)
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
vs.stop()
  




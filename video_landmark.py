from imutils import face_utils
from imutils.video import VideoStream
import imutils
import time
import argparse
import numpy
import dlib
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-p','--shape-predictor',required=True,help='path to shape predictor')
parser.add_argument('-r','--picamera',type=int,default=-1,help='path to camera')
args = vars(parser.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

vs = VideoStream(usePiCamera=args['picamera'] > 0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame,width = 400)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray,0)

    for rect in rects:
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)

        for (x,y) in shape:
            cv2.circle(frame,(x,y),1,(0,0,255),-1)
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
vs.stop()

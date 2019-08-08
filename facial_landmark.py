from imutils import face_utils #used for making translations from dlib output formatting to opencv formatting easier
import imutils #library provided by Adrian Rosebrock, creator of pyimagesearch

import argparse
import numpy
import dlib
import cv2

#parse the command-line arguments specifying the path of shape predictor and input image
parser = argparse.ArgumentParser()
parser.add_argument('-p','--shape-predictor',required = True, help = 'path to shape predictor')
parser.add_argument('-i','--image', required = True, help = 'path to input image')
#creates a dictionary with the type as the key and the file in the path as the value
args = vars(parser.parse_args())

#initialize the face detector and landmark predictor. Note how both use machine learning approaches provided by dlib library
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

#load the image to be used, then pre-process it into grayscale so that pixel intensities are easier to see
image = cv2.imread(args['image'])
image = imutils.resize(image, width = 500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#Call the detector instance using the gray image, then store the detected boxes into variable rects. Only one image pyramid will be used.
rects = detector(gray,1)

for (i, rect) in enumerate(rects):
    shape = predictor(gray,rect)
    shape = face_utils.shape_to_np(shape)

    (x,y,w,h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(image, 'Box#{}'.format(i+1), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    for (x,y) in shape:
        cv2.circle(image,(x,y), 0, (0,0,255), -1)
cv2.imshow('Output',image)
cv2.waitKey(0)


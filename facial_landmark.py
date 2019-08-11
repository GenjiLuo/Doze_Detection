from imutils import face_utils
import imutils
import argparse
import numpy
import dlib
import cv2

import face_regions #contains the required packages and the face region detection


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

    for (name, (j,k)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        clone = image.copy()
        regional_pts = shape[j:k]
        #the bounding box containing face region
        (x,y,w,h) = cv2.boundingRect(numpy.array(regional_pts))
        region = image[y:y+h, x:x+w]
        region = imutils.resize(region,width=250)
        
        cv2.putText(clone, name, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)

        for (x,y) in regional_pts:
            cv2.circle(clone,(x,y),0,(0,0,255),-1)

        cv2.imshow('clone',clone)
        cv2.imshow('region',region)
        cv2.waitKey(0)
    output = face_regions.visualize_facial_landmark(image,shape)
    cv2.imshow('Whole face',output) 
    cv2.waitKey(0)

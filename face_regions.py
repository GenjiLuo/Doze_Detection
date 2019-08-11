from imutils import face_utils
import cv2

'''Parameters:
    --image: the image you want to use for detecting face regions
    --shape: the 68 landmarks predicted after getting the image passed in to the predictor
    --colors: the color designated to each face region, formatted as a list of BGR tuples
    --alpha: the opacity of the overlay used for showing the salient regions colored
'''
def visualize_facial_landmark(image, shape, colors = None, alpha = 0.7):
    if colors == None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220)] 
    
    #create a copy of the image as an overlay and as an output, both of which will be added over each other.
    output = image.copy()
    overlay = image.copy()

    for (i, name) in enumerate(face_utils.FACIAL_LANDMARKS_IDXS.keys()):
        (j, k) = face_utils.FACIAL_LANDMARKS_IDXS[name]
        regional_pts = shape[j:k]

        if name == 'jaw':
            for l in range(1,len(regional_pts)):
                pt1 = tuple(regional_pts[l-1])
                pt2 = tuple(regional_pts[l])
                cv2.line(overlay,pt1,pt2,colors[i],2)
        else:
            hull = cv2.convexHull(regional_pts)
            cv2.drawContours(overlay,[hull],-1,colors[i],-1)
    cv2.addWeighted(overlay,alpha,output,1-alpha,0,output)
    return output

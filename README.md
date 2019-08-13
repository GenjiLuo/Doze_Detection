# Doze_Detection
## Concept/Purpose:
Sleep-deprived drivers pose a dangerous risk to themselves and others they share the road with. As a countermeasure
to this problem, I am going to be making a device that uses a camera to detect fatigue from a person's eyes and gives
off an alarm to wake them up. 
**Note**: This project was originally intended as a way of helping my mom with drowsiness while she is driving, as well as a learning experience for me getting into the details of computer vision techniques. Though the code for this already exists, I use the tutorial as a means of learning and applying to future personal projects.

## Process:
The process of building the eye detection tool will follow the same steps used by Adrian Rosebrock, the programmer who runs pyimagesearch.com. Credit for the imutils helper library used throughout this process also goes to Adrian Rosebrock. The steps will roughly follow the outline below:

1. Download/install the dlib and OpenCV library with python bindings.
  1. First, install CMake and an accepted C++ Compiler. If you are using Ubuntu, then run the command `sudo apt-get install build-essential cmake' to get make, C++ compiler, and cmake
  2. Install pip, the package manager for Python libraries, if you aren't already using it. Download it from https://bootstrap.pypa.io/get-pip.py, then run `python get-pip.py` to install.
  3. Run `pip install [package name]`, where `[package name]` is `dlib`, `opencv-contrib-python`, `numpy`, or any of the other module/package that is mentioned in the program run.
  *Note:* There is a chance that pip will not be able to install dlib if you are running windows. To fix this, download the latest version of Anaconda for python, then try running `pip install dlib` again.
2. Utilize the facial landmark detection feature of dlib alongside the image processing tools from OpenCV.
3. Utilize OpenCV's real-time video stream features to get facial detection in video camera.
4. Focus on the localized eye region for determining the rate at which the eyes blink.
5. Extend on the eye blink detection program to allow for drowsiness detection as well.
5. Implement the code onto a Raspberry Pi 3.

## The Mechanism of Facial Landmark Detection:
There are two steps so far on how facial landmarks are detected on a given input image.

1. A bounding box containing the face is detected using any of a variety of computer vision algorithsm, such as HOG + LSVM, Haar Cascades, or even deep-learning algorithms.
  1. For this project, we are using the pre-trained Histogram of Oriented Gradients + Linear Support Vector Machine provided by the dlib library for the task of face detection.
2. Given a bounding box containing the face, we use a shape predictor provided by the dlib library based on research done on [facial alignment](http://www.csc.kth.se/~vahidk/face_ert.html) using gradient boosting.

## Where We Are Now:

I wrote out the code/ procedure I have learned so far on how a face detector and landmark predictor are used to show localized areas of the face. I am also using the 68 point iBUG 300-W dataset which the dlib facial landmark predictor was trained on, which I now include in the top directory. I then applied the same procedure of face landmark detection to frames in a video rather than on a single image. Now, I used the eye aspect ratio (EAR) of the left and right eye landmarks to designate whether my eyes are open or closed, depending on which side of the EAR threshold they are on, and I calculate from that the amount of times that the person has blinked so far.

## How to Use:

1. Git clone the project onto your local directory.
2. cd into the directory you cloned the repository.
3. There are four different programs you can run:
	1. To detect faces in images, run `python facial_landmark.py --shape-predictor face_landmarks_68.dat --image [path of jpg/png of image you want]`. You can find samples of pictures in the pictures directory. Alternatively, you can use -p and -i as shorthand for --shape-predictor and --image, respectively.
	2. To detect faces in framew, run 'python video_landmark.py --shape-predictor face_landmarks_68.dat [--picamera 1]` with the `[]` being optional if you want to run the camera through your raspberry pi instead of your webcam. To quit the program, press q.
	3. To detect blinks on a face in camera, run `python blink_detection.py --shape-predictor face_landmarks_68.dat`. Once the window opens, you can blink and confirm the blink count to see that the threshold has been set correctly for you. If not, go into blink_detection.py, and adjust the EYE_RATIO_THRESHOLD accordingly.
	4. To detect signs of drowsiness, run `python doze_detection.py --shape-predictor face_landmarks_68.dat --alarm [path of mp3 or wav file of alarm you want]`. You can find a sample in the sounds directory. Try to use sound files that are a few seconds long, the program stops while the sound plays. Face the camera, and if you are dozing off, the window will sound the alarm and give a warning message. **Note**: Keep in mind that the angle at which you look at the camera will affect the eye ratio calculated. In such a case, the alarm may go off at a false positive.

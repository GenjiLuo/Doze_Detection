# Fatigue_Detection
## Concept/Purpose:
Sleep-deprived drivers pose a dangerous risk to themselves and others they share the road with. As a countermeasure
to this problem, I am going to be making a device that uses a camera to detect fatigue from a person's eyes and gives
off an alarm to wake them up. 
**Update** This project was originally intended as a way of helping my mom with drowsiness while she is driving, as well as a learning experience for me getting into the details of computer vision techniques. Though the code for this already exists, I use the tutorial as a means of learning and applying to future personal projects.

## Process:
The process of building the eye detection tool will follow the same steps used by Adrian Rosebrock, the programmer who runs pyimagesearch.com. The steps will roughly follow the outline below:

1. Download/install the dlib and OpenCV library with python bindings.
  1. First, install CMake and an accepted C++ Compiler. If you are using Ubuntu, then run the command `sudo apt-get install build-essential cmake' to get make, C++ compiler, and cmake
  2. Install pip, the package manager for Python libraries, if you aren't already using it. Download it from https://bootstrap.pypa.io/get-pip.py, then run `python get-pip.py` to install.
  3. Run `pip install [package name]`, where `[package name]` is `dlib` or `opencv-contrib-python`.
  *Note:* There is a chance that pip will not be able to install dlib if you are running windows. To fix this, download the latest version of Anaconda for python.
2. Utilize the facial landmark detection feature of dlib alongside the image processing tools from OpenCV
3. Utilize OpenCV's real-time features to get facial detection in video camera.
4. Focus on the localized eye region for determining the rate at which the eyes blink.
5. Implement the code onto a Raspberry Pi 3.

## The Mechanism of Facial Landmark Detection:
There are two steps so far on how facial landmarks are detected on a given input image.

1. A bounding box containing the face is detected using any of a variety of computer vision algorithsm, such as HOG + LSVM, Haar Cascades, or even deep-learning algorithms.
  1. For this project, we are using the pre-trained Histogram of Oriented Gradients + Linear Support Vector Machine provided by the dlib library for the task of face detection.
2. Given a bounding box containing the face, we use a shape predictor provided by the dlib library based on research done on [facial alignment](http://www.csc.kth.se/~vahidk/face_ert.html) using gradient boosting.

## Where We Are Now:

I am writing out the code/ procedure I have learned so far on how a face detector and landmark predictor are used to show localized areas of the face. I am also using the 68 point iBUG 300-W dataset which the dlib facial landmark predictor was trained on, which I now include in the top directory.

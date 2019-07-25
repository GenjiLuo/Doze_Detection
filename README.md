# Fatigue_Detection
## Concept:
Sleep-deprived drivers pose a dangerous risk to themselves and others they share the road with. As a countermeasure
to this problem, I am going to be making a device that uses a camera to detect fatigue from a person's eyes and gives
off an alarm to wake them up.
## Process:
This project will most likely go through several changes as the development process progresses. Here is the roadmap
thus far:
1. Download/install the dlib and opencv library with python bindings.
  1. First, install CMake and an accepted C++ Compiler
  2. Get pip to streamline the process of installing CMake. 
  2.5. **Note:** There is a chance that pip will not be able to
  install dlib. To fix this, download the latest version of Anaconda for python.
  3. Run `pip install [package name]`, where `[package name]` is `dlib` or `opencv-contrib-python`.
2. Utilize the facial landmark detection feature of dlib.
3. Utilize OpenCV's real-time features to get facial detection in video camera.
4. Focus on the localized eye region for determining the rate at which the eyes blink.
5. Implement the code onto a Raspberry Pi 3.
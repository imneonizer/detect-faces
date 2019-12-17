## Python Face Detection

A lot of people are very new to computer vision and at first sight all they wanna do (including me) is to detect faces in images/videos.

while there are lot's of tutorials and guides are available online most of them lacks proper instructions or unmet dependencies.

so i have provided this small program which is basically a face detector class, you can simply import and use. And the cool thing is you don't have to worry about tuning model weight file path you can import it from any directory and it will automatically load it's required weight files from the same directory.

**steps to run the code:**

````python
python detect_faces.py
````

That's it and you will see a window pop up with detecting faces in the current webcam feed. Obviously you'll need to have a webcam attached to your computer.

The code for face detection is very simple:

````python
import face_detection_model as fd
import numpy as np
import cv2

#video capture object
cap = cv2.VideoCapture(0)

while True:
    #read latest frame
    success, frame = cap.read()
    if not success: break

    #flip horizontal webcam frame
    frame = cv2.flip(frame,1)

    #iterate through all detected faces
    for x1,y1,x2,y2,center in fd.detector.detect(frame):
        #draw rectange on detected faces
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)

    #show live bbox drawn stream
    cv2.imshow('camera', frame)
    if cv2.waitKey(1) == ord('q'): break
````

You can basically replace ``0`` with any video file path in the function ``cv2.VideoCapture(0)`` to run the face detection on videos. Also make sure you python environment has both ``OpenCV`` and ``Numpy`` installed  before running the code.

****
Original Resource: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

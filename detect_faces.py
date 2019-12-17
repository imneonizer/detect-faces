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

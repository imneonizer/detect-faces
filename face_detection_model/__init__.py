import numpy as np
import cv2
import os

model_path = os.path.dirname(os.path.realpath(__file__))

class FaceDetector():
    def __init__(self, confidence=0.5):
        proto = model_path+'/deploy.prototxt.txt'
        model = model_path+'/res10_300x300_ssd_iter_140000.caffemodel'
        self.net = cv2.dnn.readNetFromCaffe(proto, model)
        self.confidence = confidence

    def detect(self, frame):
        # construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        image = frame
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        # loop over the detections
        all_detection = []
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            model_confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if model_confidence > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

                conf = round((model_confidence * 100),2)
                (x1,y1,x2,y2) = box.astype("int")
                center = (x1+x2)//2, (y1+y2)//2
                all_detection.append((x1,y1,x2,y2, center))

        return all_detection

detector = FaceDetector()

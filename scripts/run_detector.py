
"""
    Run the object detector using a Webcam
"""

import cv2

from detector import Detector

import pdb

def main():
    """
    Main function for running the object detector
    """

    # Initialize the object detector
    detector = Detector("/detector/models/peoplenet/model.trt")

    # # Load image
    # frame = cv2.imread("/detector/test.jpg")
    # # convert to rgb
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # anno_frame = detector.detect(frame)
    # cv2.imwrite("/detector/test_anno.jpg", anno_frame)

    # Initialize webcam video stream
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect objects in the frame and return annotated detections
        # frame = detector.detect(frame)
        frame = detector.detect(frame)

        # Resize frame to fit the screen
        new_shape = (int(frame.shape[1]*3), int(frame.shape[0]*3))
        frame = cv2.resize(frame, new_shape)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()

if __name__ == "__main__":
    main()
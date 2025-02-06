
"""
    Run the object detector using a Webcam
"""

import cv2

import pdb

def main():
    """
    Main function for running the object detector
    """


    # Initialize webcam video stream
    cap = cv2.VideoCapture(1)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()

if __name__ == "__main__":
    main()
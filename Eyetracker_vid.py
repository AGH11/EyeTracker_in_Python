import cv2 as cv
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector

# Indices of the left eye landmarks
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Initialize face and face mesh detectors
detector = FaceDetector()
meshdetector = FaceMeshDetector(maxFaces=1)

# Open a video capture object
# Use a webcam
cap = cv.VideoCapture(0)
# Use a video
# cap = cv.VideoCapture('Video address')
if not cap.isOpened():
    print('Error opening video capture')

while cap.isOpened():
    # Read frames from the video capture
    ret, frame = cap.read()
    ret, frame2 = cap.read()

    if ret:
        # Detect faces in the frame
        face_img, bbox = detector.findFaces(frame)
        # Detect face mesh in the frame
        face_img, faces = meshdetector.findFaceMesh(frame)

        if bbox:
            center = bbox[0]["center"]   

            if faces:
                # Extract left eye points
                left_eye_points = np.array([[faces[0][p][0], faces[0][p][1]] for p in LEFT_EYE])
                (ex, ey, ew, eh) = cv.boundingRect(left_eye_points)
                
                # Extract the region of interest (ROI) for the eye
                eye_roi = frame2[ey:ey + eh, ex:ex + ew]
                eye_roi_gr = cv.cvtColor(eye_roi, cv.COLOR_BGR2GRAY)
                
                # Threshold the eye region
                _, iris = cv.threshold(eye_roi_gr, 40, 255, cv.THRESH_BINARY_INV)
                
                # Find contours in the iris region
                contours, _ = cv.findContours(iris, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

                if contours:
                    (ix, iy, iw, ih) = cv.boundingRect(contours[0])
                    ix_center, iy_center = ix + int(iw/2) + ex, iy + int(ih/2) + ey
                    
                    # Draw a circle at the center of the iris
                    cv.circle(frame2, (ix_center, iy_center), 5, (0, 0, 255), -1)

                    ix_center_e, iy_center_e = ix + int(iw/2), iy + int(ih/2)

                    offset = 10
                    # Determine the direction of gaze based on the position of the iris center
                    if ix_center_e > int(ew/2) + offset:
                        text = "right"
                    elif ix_center_e < int(ew/2) - offset:
                        text = "left"
                    else:
                        text = "center"

                    # Display the direction of gaze
                    cv.putText(frame2, text, (100, 100), cv.FONT_HERSHEY_PLAIN, 3, (0, 60, 0), 2)

        # Display the processed frame
        cv.imshow('frame2', frame2)

        # Break the loop if 'q' key is pressed
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object and close all windows
cap.release()
cv.destroyAllWindows()
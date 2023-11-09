import cv2 as cv
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector

# Indices of the left eye landmarks
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Initialize face and face mesh detectors
detector = FaceDetector()
meshdetector = FaceMeshDetector(maxFaces=1)

# Load the face image
face_img = cv.imread('data/face.jpg')
face_img2 = face_img.copy()

# Detect faces in the face image
face_img, bbox = detector.findFaces(face_img)
# Detect face mesh in the face image
face_img, faces = meshdetector.findFaceMesh(face_img)

if bbox:
    center = bbox[0]['center']
    if faces:
        # Extract left eye points
        left_eye_points = np.array([[faces[0][p][0], faces[0][p][1]] for p in LEFT_EYE])
        (ex, ey, ew, eh) = cv.boundingRect(left_eye_points)
        
        # Extract the region of interest (ROI) for the eye
        eye_roi = face_img2[ey:ey+eh, ex:ex+ew]
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
            cv.circle(face_img2, (ix_center, iy_center), 5, (0, 0, 255), -1)

            ix_center_e, iy_center_e = ix + int(iw/2), iy + int(ih/2)
            # Determine the direction of gaze based on the position of the iris center
            if ix_center_e > int(ew/2):
                print('right')
            elif ix_center_e < int(ew/2):
                print('left')

# Display the processed face image
cv.imshow('eye_roi', face_img2)
cv.waitKey(0)
cv.destroyAllWindows()
import cv2
import numpy as np
import dlib

# Load the face detector and landmark predictor models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the image
img = cv2.imread("face.jpg")

# Detect the face in the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 0)

# Loop over the face detections
for (i, rect) in enumerate(rects):
    # Get the facial landmarks for the face region
    shape = predictor(gray, rect)
    shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

    # Create a mask for the mouth region
    mouth_mask = np.zeros_like(gray)
    mouth_mask = cv2.fillPoly(mouth_mask, [shape[48:60]], 255)

    # Extract the mouth region from the image
    mouth = cv2.bitwise_and(img, img, mask=mouth_mask)

    # Apply a random color to the mouth region
    color = np.random.randint(0, 256, size=(3,))
    mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2HSV)
    mouth[:, :, 0] += color[0]
    mouth[:, :, 1] += color[1]
    mouth[:, :, 2] += color[2]
    mouth = cv2.cvtColor(mouth, cv2.COLOR_HSV2BGR)

    # Blend the modified mouth region with the original image
    img = cv2.addWeighted(img, 0.8, mouth, 0.2, 0)

# Display the modified image
cv2.imshow("Funny Face", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

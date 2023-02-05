import cv2
import numpy as np
import time

# Load the cascade classifier files
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Wait for the camera to warm up
time.sleep(2)

start_time = time.time()
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop over the faces
    for (x, y, w, h) in faces:
        # Crop the face
        face_roi = gray[y:y+h, x:x+w]

        # Detect smiles in the face
        smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=22)

        # Loop over the smiles
        for (sx, sy, sw, sh) in smiles:
            # Draw a rectangle around the smile
            cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (255, 0, 0), 2)
            cv2.putText(frame, 'Smiling', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
    # Display the frame
    cv2.imshow('Smile Detection', frame)

    # Check if 3 seconds have passed
    if time.time() - start_time > 3:
        break

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the output image
cv2.imwrite("output_image.jpg", frame)

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()

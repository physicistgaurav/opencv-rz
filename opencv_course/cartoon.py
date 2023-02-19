import cv2

# Create a VideoCapture object to capture frames from the camera
cap = cv2.VideoCapture(0)

while True:
    # Read frames from the camera
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filtering
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(bilateral, 50, 150)

    # Apply threshold to create a binary mask
    ret, mask = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY_INV)

    # Apply bitwise-and operation to create cartoon effect
    cartoon = cv2.bitwise_and(frame, frame, mask=mask)

    # Display output image
    cv2.imshow('Cartoon Face', cartoon)

    # Check for user input to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()

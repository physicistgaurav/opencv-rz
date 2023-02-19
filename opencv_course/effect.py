import cv2

# Define the function to apply the effect
def apply_effect(frame, effect):
    if effect == 'grayscale':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif effect == 'negative':
        return 255 - frame
    elif effect == 'blur':
        return cv2.GaussianBlur(frame, (15, 15), 0)
    else:
        return frame

# Open the camera and start capturing video
cap = cv2.VideoCapture(0)

# Initialize the effect to none
effect = 'none'

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
        # Apply the selected effect to the frame
        if effect != 'none':
            frame = apply_effect(frame, effect)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Check for key press and update the effect
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('g'):
            effect = 'grayscale'
        elif key == ord('n'):
            effect = 'negative'
        elif key == ord('b'):
            effect = 'blur'
        elif key == ord('c'):
            effect = 'none'

    else:
        break

# Release everything and close the windows
cap.release()
cv2.destroyAllWindows()

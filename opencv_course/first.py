import cv2
import matplotlib.pyplot as plt

ch_image =cv2.imread('ch.jpg')
coke_img = cv2.imread('coke.jpg')

#matplotlib imshow

plt.imshow(ch_image)
plt.title("matplotlib_imshow")
plt.show()

#useopencv imshow()

window1= cv2.namedWindow('w1')
cv2.imshow(window1, ch_image)
cv2.waitKey(8000)
cv2.destroyWindow(window1)

window2= cv2.namedWindow('w2')
cv2.imshow(window2, coke_img)
cv2.waitKey(8000)
cv2.destroyWindow(window2)

#use opencv imshow() display until any key is pressed

window3= cv2.namedWindow('w3')
cv2.imshow(window3, ch_image)
cv2.waitKey(0)
cv2.destroyWindow(window3)

window4= cv2.namedWindow('w4')

Alive = True
while Alive:
    cv2.imshow(window4, coke_img)
    keypress = cv2.waitKey(1)
    if keypress == ord('q'):
        Alive= False
cv2.destroyAllWindows()
stop =1

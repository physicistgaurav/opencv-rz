import cv2
import numpy as np
from PyPDF2 import PdfFileWriter, PdfFileReader

# read input image
img = cv2.imread("ava.jpg")

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply adaptive thresholding to obtain binary image
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# find contour with maximum area
max_area = 0
best_cnt = None
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > max_area:
        max_area = area
        best_cnt = cnt

# create mask and draw contour on it
mask = np.zeros(img.shape[:2], np.uint8)
cv2.drawContours(mask, [best_cnt], 0, 255, -1)
cv2.bitwise_and(img, img, mask=mask)

# find rotated rectangle
rect = cv2.minAreaRect(best_cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)

# apply perspective transform to obtain top-down view
width = int(rect[1][0])
height = int(rect[1][1])
src_pts = box.astype("float32")
dst_pts = np.array([[0, height-1],
                    [0, 0],
                    [width-1, 0],
                    [width-1, height-1]], dtype="float32")
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
dst = cv2.warpPerspective(img, M, (width, height))

# save output image as pdf file
pdf_writer = PdfFileWriter()
pdf_writer.addPage(PdfFileReader(open("output.pdf", "rb")).getPage(0))
cv2.imwrite("temp.jpg", dst)
pdf_writer.addPage(PdfFileReader(open("temp.jpg", "rb")).getPage(0))
with open("output.pdf", "wb") as f:
    pdf_writer.write(f)

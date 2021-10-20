import cv2

image = cv2.imread('../image/TestTrack_1.png', 1)
image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)

cv2.imshow("result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



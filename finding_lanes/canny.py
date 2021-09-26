import cv2
import numpy as np

image = cv2.imread('../image/TestTrack_1.png', 1)
image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)

lane_image = np.copy(image)
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY) # 3 Channel -> 1 Channel
blur = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blur, 50, 400)

cv2.imshow("result", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
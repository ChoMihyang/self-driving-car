import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # 3 Channel -> 1 Channel
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 400)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(-350, height), (850, height), (320, 110)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

image = cv2.imread('../image/TestTrack_1.png', 1)
image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)

lane_image = np.copy(image)
canny = canny(lane_image)
cropped_image = region_of_interest(canny)
cv2.imshow("result", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
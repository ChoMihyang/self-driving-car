import cv2
import numpy as np

def region_of_interest(image):
    h = int(image.shape[0])
    w = int(image.shape[1])
    
    # 관심 영역 표시 [LB][LT][RT][RB]
    _shape = np.array([
        [int(0.15 * w), int(0.95 * h)], [int(0.15 * w), int(0.65 * h)],
        [int(0.80 * w), int(0.65 * h)], [int(0.80 * w), int(0.95 * h)],
    ])

    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    roi_image = cv2.bitwise_and(image, mask)

    return roi_image

img = cv2.imread('../image/stop_line_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
roi_img = region_of_interest(gray)

ret, thresh = cv2.threshold(roi_img, 127, 255, 0)

contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour_image = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

cv2.imshow('image', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

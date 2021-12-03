import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 10, 230, None, 3)
    return canny

def region_of_interest(image):
    h = int(image.shape[0])
    w = int(image.shape[1])
    
    # 관심 영역 표시 [LB][LT][RT][RB]
    _shape = np.array([
        [int(0.01 * w), int(0.95 * h)], [int(0.01 * w), int(0.65 * h)],
        [int(0.99 * w), int(0.65 * h)], [int(0.99 * w), int(0.95 * h)],
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

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3) # 이미지, 시작 좌표, 끝 좌표, 색깔, 굵기
    return line_image

# 이미지 불러오기
img = cv2.imread('../image/stop_line_1.jpg')
# img = cv2.imread('../image/stop_line_2.jpg')

while cv2.waitKey(1) != ord('q'):

    lane_image = np.copy(img)
    canny_img = canny(lane_image)
    roi_image = region_of_interest(canny_img)

    lines = cv2.HoughLinesP(roi_image, 1, np.pi / 100, 70, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = display_lines(lane_image, lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    
    cv2.imshow('result', combo_image)

cv2.destroyAllWindows()

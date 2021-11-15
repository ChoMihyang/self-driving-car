import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 10, 300, None, 3)
    return canny

def region_of_interest(image):
    h = int(image.shape[0])
    w = int(image.shape[1])

    # 관심 영역 표시 [LB][LT][RT][RB]
    # img_1
    # _shape = np.array([
    #     [int(0.11 * w), int(0.65 * h)], [int(0.13 * w), int(0.45 * h)],
    #     [int(0.75 * w), int(0.45 * h)], [int(0.77 * w), int(0.65 * h)],
    # ])

    # img_2
    # _shape = np.array([
    #     [int(0.01 * w), int(0.95 * h)], [int(0.01 * w), int(0.75 * h)],
    #     [int(0.90 * w), int(0.75 * h)], [int(0.90 * w), int(0.95 * h)],
    # ])

    # img_3
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
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image

# img = cv2.imread('../image/stop_line_1.png')
# img = cv2.imread('../image/stop_line_2.png')
img = cv2.imread('../image/stop_line_3.png')

while cv2.waitKey(1) != ord('q'):

    lane_image = np.copy(img)
    canny_img = canny(lane_image)
    roi_image = region_of_interest(canny_img)

    lines = cv2.HoughLinesP(roi_image, 1, np.pi / 2, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = display_lines(lane_image, lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    
    cv2.imshow('result', canny_img)

cv2.destroyAllWindows()

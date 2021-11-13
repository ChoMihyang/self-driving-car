import cv2
import numpy as np

# 이미지 프로세싱
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 200, None, 3)
    return canny

# 관심 영역 표시
def region_of_interest(image):
    h = int(image.shape[0])
    w = int(image.shape[1])

    # 관심 영역 표시 [LB][LT][RT][RB]
    _shape = np.array([
        [int(0.11 * w), int(0.65 * h)], [int(0.13 * w), int(0.45 * h)],
        [int(0.75 * w), int(0.45 * h)], [int(0.77 * w), int(0.65 * h)],
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

# 선 그리기
def display_lines(image, lines):
    global count
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # print('x1 : ', x1, ' / y1 : ', y1, ' / x2 : ', x2, ' / y2 : ', y2)
            # line(img, 시작 좌표(x1, y1), 종료 좌표(x2, y2), color, 선 두께)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5) 
    
    return line_image


##### 정지선 카운팅 메서드 시작 #####
stop_line_count = 0

def flag():
    global stop_line_count

    return stop_line_count   
##### 정지선 카운팅 메서드 끝 #####


# 비디오 불러오기
cap = cv2.VideoCapture("../image/stopline_test_1.mp4")

while True:
    ret, img = cap.read()

    if not ret:
        break

    lane_image = np.copy(img)
    canny_img = canny(lane_image)
    roi_image = region_of_interest(canny_img)
    # TODO : 정지 (수평)선 기울기에 따라 𝜃값 변경(0~180)
    # HoughLinesP(이미지, rho[0~1], theta[𝜃], 임계값, 선의 최소 길이, 선사이 최대 허용 간격)
    lines = cv2.HoughLinesP(roi_image, 1, np.pi / 2, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = display_lines(lane_image, lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
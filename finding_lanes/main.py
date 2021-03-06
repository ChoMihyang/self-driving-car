import math
import scipy.fftpack 
import numpy as np
import cv2

# 좌표 설정 
_LB = 0    # 좌하
_LT = 1    # 좌상
_RB = 2    # 우하
_RT = 3    # 우상

# 색상 설정
_RED = (255, 0, 0)
_GREEN = (0, 255, 0)
_BLUE = (0, 0, 255)
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)

# TODO : ?
_PAUSE_TIME = 0.01
_DEG_ERROR_RANGE = 1
_DIST_ERROR_RANGE = 8

# 영상 이미지 상 좌표 표시하기
def make_source_marker(image):
    marked_img = image.copy()
    marker_size = 10
    # LB, LT, RB, RT
    position = [
        (60, 350), (170, 80), (630, 350), (510, 80)
    ]
    cv2.circle(marked_img, position[_LB], marker_size, _RED, -1)
    cv2.circle(marked_img, position[_LT], marker_size, _GREEN, -1)
    cv2.circle(marked_img, position[_RB], marker_size, _BLUE, -1)
    cv2.circle(marked_img, position[_RT], marker_size, _BLACK, -1)

    return marked_img, position

# 영상 이미지 변형 (bird eyes view)
def wrapping_img(image, source_position):
    (h, w) = (image.shape[0], image.shape[1])

    source = np.float32(source_position)
    destination = np.float32([(60, 380), (60, 50), (630, 380), (630, 50)])

    transform_matrix = cv2.getPerspectiveTransform(source, destination)     # Matrix to wrap the image for birdseye window
    minverse = cv2.getPerspectiveTransform(destination, source)             # Inverse matrix to unwrap the image for final window
    wrapped_img = cv2.warpPerspective(image, transform_matrix, (w, h))

    return wrapped_img, minverse

# 이미지 프로세싱 (회색조 변환, 블러 처리, 임계 처리, canny)    
def color_filtering_img(image, canny_sigma, low_thresh, high_thresh):
    g_blur_size = 5
    m_blur_size = 5
    thresh = 170
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    g_blur_img = cv2.GaussianBlur(image, (g_blur_size, g_blur_size), canny_sigma)
    m_blur_img = cv2.medianBlur(g_blur_img, m_blur_size)
    ret, thr_img = cv2.threshold(m_blur_img, thresh, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(thr_img, low_thresh, high_thresh)

    # cv2.imshow("gau", g_blur_img)
    # cv2.imshow("med", m_blur_img)

    return canny

# 차선 검출 관심 영역 설정
def set_roi_area(image):
    x = int(image.shape[1])
    y = int(image.shape[0])

    # 한 붓 그리기
    _shape = np.array([
        [int(0.05 * x), int(0.9 * y)], [int(0.05 * x), int(0.1 * y)],
        [int(0.4 * x), int(0.1 * y)], [int(0.4 * x), int(0.9 * y)],
        [int(0.6 * x), int(0.9 * y)], [int(0.6 * x), int(0.1 * y)],
        [int(0.95 * x), int(0.1 * y)], [int(0.95 * x), int(0.9 * y)],
        [int(0.05 * x), int(0.9 * y)]
    ])

    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_roi_image = cv2.bitwise_and(image, mask)

    return masked_roi_image

# histogram() : 영상 이미지의 밝기의 분포를 그래프로 표현, 이미지 전체의 밝기 분포와 채도를 파악
# 이진화된 이미지는 하나의 채널과 0~255로 이루어진 이미지
# -> 차선이 있는 부분 - 255에 근접, 차선이 아닌 부분 - 0에 근접
def plot_histogram(image):
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    mid_point = np.int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:mid_point])                # histogram 좌표의 왼 편에서 수치가 가장 높은 부분 = 왼쪽 차선이 있음을 인식 
    right_base = np.argmax(histogram[mid_point:]) + mid_point   # histogram 좌표의 오른 편에서 수치가 가장 높은 부분 = 오른쪽 차선이 있음을 인식

    # 차선을 인식한 부분(프레임?)의 좌표 반환
    return left_base, right_base

# 슬라이드 윈도우 기반 차선 인식
# param - image : roi 이미지, left_current : 왼쪽 차선 영역, right_current : 오른쪽 차선 영역
def slide_window_search(image, left_current, right_current):
    # out_img :
    #   [[0 0 0]
    #    [0 0 0]
    #    [0 0 0]
    #    ...
    #   [0 0 0]
    #   [0 0 0]
    #   [0 0 0]]
    out_img = np.dstack((image, image, image))
    nwindows = 4 # 순환할 윈도우 영역 수 
    window_height = np.int(image.shape[0] / nwindows) # 윈도우 영역 분할
    margin = 100
    minpix = 50
    thickness = 2

    nonzero = image.nonzero()           # nonzero() : 요소들 중 0이 아닌 값들의 index반환, 선이 있는(= 흰 픽셀이 존재하는) 부분의 '인덱스'만 저장
    # len(nonzero[0]) : 1296 = ??픽셀에 담긴 값 중 0이 아닌 값의 개수, 1296개의 흰 픽셀이 존재
    nonzero_y = np.array(nonzero[0])    # 선이 있는(= 흰 픽셀이 존재하는) 부분 y의 인덱스 값
    nonzero_x = np.array(nonzero[1])    # 선이 있는(= 흰 픽셀이 존재하는) 부분 x의 인덱스 값
    
    left_lane = []
    right_lane = []
    
    # windows 수만큼 반복하며 차선 찾기
    for w in range(nwindows):

        # 사각형의 시작점과 종료점 좌표 구하기
        win_y_low = image.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = image.shape[0] - w * window_height  # window 아랫 부분
        win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
        win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
        win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위
        win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래

        # rectangle - 사각형 그리기 
        # param - 이미지 파일, 시작점 좌표(x, y), 종료점 좌표(x, y), 색상, 선 두께
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), _GREEN, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), _GREEN, thickness)
        
        # window 사이즈를 만족하는 픽셀 추려내기 (window 영역 안에 들어오는지 확인)
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (
                nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
                nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)

        # TODO : minpix (50) - 최소 픽셀? 기준은?
        if len(good_left) > minpix:
            left_current = np.int(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int(np.mean(nonzero_x[good_right]))

    left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
    right_lane = np.concatenate(right_lane)

    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]

    left_fit = np.polyfit(lefty, leftx, 2)      # polyfit() -> 2차 함수 그래프로 차선 그리기
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ltx = np.trunc(left_fitx)   # np.trunc 소수점 버림
    rtx = np.trunc(right_fitx)

    out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = _RED
    out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = _BLUE

    ret = {'left_fitx': ltx, 'right_fitx': rtx, 'ploty': ploty}

    return ret

def draw_lane_lines(original_image, warped_image, minv, draw_info):
    left_fitx, right_fitx, ploty = draw_info['left_fitx'], draw_info['right_fitx'], draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), _WHITE)

    # todo test code
    center = np.squeeze(np.int_([pts_mean]))
    start, end = center[-1], center[0]
    arr = [start[0], start[1], end[0], end[1]]

    # 방향 각도 계산
    rad = math.atan2(arr[3] - arr[1], arr[2] - arr[0])
    deg = int((rad * 180) / math.pi - 90)

    # 곡률 거리 계산
    mid1 = np.int_([(start[0] + end[0]) / 2, (start[1] + end[1]) / 2])
    mid2 = np.squeeze(np.int_([pts_mean]))[202]

    x = (mid1[0] - mid2[0]) ** 2
    y = (mid1[1] - mid2[1]) ** 2
    dist = int((x + y) ** 0.5)

    cv2.circle(color_warp, mid1, 5, _BLACK, -1)
    cv2.circle(color_warp, mid2, 5, _GREEN, -1)
    cv2.circle(color_warp, start, 10, _RED, -1)
    cv2.circle(color_warp, end, 10, _BLACK, -1)

    new_warp = cv2.warpPerspective(color_warp, minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, new_warp, 0.4, 0)

    return pts_mean, result, deg, dist   

#### imclearborder definition
def imclearborder(imgBW, radius):

    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]    

    contourList = [] # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

#### bwareaopen definition
def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

def homomorphic_filtering(img, sigma):

    # Convert image to 0 to 1, then do log(1 + I)
    imgLog = np.log1p(np.array(img, dtype="float") / 255)

    # Create Gaussian mask of sigma = 10
    M = 2 * rows + 1
    N = 2 * cols + 1
    (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
    centerX = np.ceil(N / 2)
    centerY = np.ceil(M / 2)
    gaussianNumerator = (X - centerX) ** 2 + (Y - centerY) ** 2

    # Low pass and high pass filters
    Hlow = np.exp(-gaussianNumerator / (2 * sigma * sigma))
    Hhigh = 1 - Hlow

    # Move origin of filters so that it's at the top left corner to
    # match with the input image
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    # Filter the image and crop
    If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))

    # Set scaling factors and add
    gamma1 = 0
    gamma2 = 1
    Iout = gamma1 * Ioutlow[0:rows, 0:cols] + gamma2 * Iouthigh[0:rows, 0:cols]
    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255 * Ihmf, dtype = "uint8")

    # Threshold the image - Anything below intensity 65 gets set to white
    Ithresh = Ihmf2 < 65
    Ithresh = 255 * Ithresh.astype("uint8")

    # Clear off the border.  Choose a border radius of 5 pixels
    Iclear = imclearborder(Ithresh, 5)

    # Eliminate regions that have areas below 120 pixels
    Iopen = bwareaopen(Iclear, 120)

    return Ihmf2, Ithresh, Iopen

# 비디오 불러오기
cap = cv2.VideoCapture("../image/Shadow_Test_1.avi")
winname = "result"

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

cv2.namedWindow(winname)

# 비디오 읽기
while True:
    ret, img = cap.read()

    if not ret:
        break
    
    # Number of rows and columns
    rows = img.shape[0]
    cols = img.shape[1]

    # Remove some columns from the beginning and end
    img = img[:, 59:cols-20]

    # Number of rows and columns
    rows = img.shape[0]
    cols = img.shape[1]

    # h : 해당 이미지(영상)의 높이, w : 해당 이미지(영상)의 너비
    # (h, w) = (img.shape[0], img.shape[1])
    homo_sigma = 10
    canny_sigma = 2
    low_thresh = 35
    high_thresh = 180

    mark_img, src_position = make_source_marker(img)
    wrap_img, minv = wrapping_img(img, src_position)
    Ihmf2, Ithresh, Iopen = homomorphic_filtering(wrap_img, homo_sigma) # homomorphic filtering
    filter_img = color_filtering_img(Ihmf2, canny_sigma, low_thresh, high_thresh) # canny filtering
    roi_img = set_roi_area(filter_img)

    left, right = plot_histogram(roi_img) # 왼쪽 차선 영역과 오른쪽 차선 영역을 구분 
    draw_info = slide_window_search(roi_img, left, right)
    mean_pts, result, deg, dist = draw_lane_lines(img, roi_img, minv, draw_info)

    dir = "LEFT" if ((deg < _DEG_ERROR_RANGE * -1) and dist > _DIST_ERROR_RANGE) \
        else ("RIGHT" if ((deg > _DEG_ERROR_RANGE) and dist > _DIST_ERROR_RANGE)
              else "FRONT")

    cv2.putText(result, f"Deg : {deg}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(result, f"Dist : {dist}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(result, f"[{dir}]", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

    # 비디오 띄우기
    cv2.imshow("result", result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
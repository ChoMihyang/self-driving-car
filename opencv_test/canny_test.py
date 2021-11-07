import cv2
import numpy as np

def canny(img, sigma, low_thresh, high_thresh):
    lane_image = np.copy(img)
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY) # 3 Channel -> 1 Channel
    blur = cv2.GaussianBlur(gray, (5, 5), sigma) # GaussianBlur(이미지, 사이즈, x/y방향 표준편차)
    canny = cv2.Canny(blur, low_thresh, high_thresh) # Canny(이미지, 낮은 경계값, 높은 경계값)  
    cv2.imshow("GaussianbBlur_image", blur)

    return lane_image, canny

src = cv2.imread("../image/shadow_test_1_1.png", 1)
src = cv2.resize(src, dsize=(700, 700), interpolation=cv2.INTER_AREA)

cv2.namedWindow("Trackbar Windows")

cv2.createTrackbar("G_sigma", "Trackbar Windows", 0, 10, lambda x : x)
cv2.setTrackbarPos("G_sigma", "Trackbar Windows", 0)

cv2.createTrackbar("C_low_thr", "Trackbar Windows", 0, 500, lambda x : x)
cv2.setTrackbarPos("C_low_thr", "Trackbar Windows", 50)

cv2.createTrackbar("C_high_thr", "Trackbar Windows", 0, 500, lambda x : x)
cv2.setTrackbarPos("C_high_thr", "Trackbar Windows", 400)


while cv2.waitKey(1) != ord('q'):

    sigma = cv2.getTrackbarPos("G_sigma", "Trackbar Windows")
    low_thresh = cv2.getTrackbarPos("C_low_thr", "Trackbar Windows")
    high_thresh = cv2.getTrackbarPos("C_high_thr", "Trackbar Windows")
    
    blur, canny_result = canny(src, sigma, low_thresh, high_thresh)
    
    cv2.imshow("Trackbar Windows", canny_result)

cv2.destroyAllWindows()
    
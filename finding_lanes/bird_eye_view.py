import cv2
import numpy as np
from numpy.matrixlib.defmatrix import matrix

cap = cv2.VideoCapture('../image/video_test_1.mp4')

while True:
    thr, frame = cap.read()

    if thr :
        # 이미지 변형 (원근감 제거)
        cv2.circle(frame, (150, 70), 5, (0, 0, 255), -1) # 좌상 - 빨
        cv2.circle(frame, (540, 70), 5, (0, 255, 255), -1) # 우상 - 노
        cv2.circle(frame, (50, 350), 5, (255, 0, 255), -1) # 좌하 - 분
        cv2.circle(frame, (700, 350), 5, (255, 255, 0), -1) # 우하 - 하늘

        pts1 = np.float32([[150, 70], [540, 70], [50, 350], [700, 350]])
        pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        result = cv2.warpPerspective(frame, matrix, (640, 480))
    
        cv2.imshow("Frame", frame)
        cv2.imshow("Perspective Transformation", result)

        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
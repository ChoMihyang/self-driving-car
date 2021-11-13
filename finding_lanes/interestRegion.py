# import cv2
# import numpy as np

# def canny(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # 3 Channel -> 1 Channel
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     canny = cv2.Canny(blur, 50, 400)
#     return canny

# def region_of_interest(image):
#     height = image.shape[0]
#     polygons = np.array([
#         [(-230, height), (700, height), (150, 310)]
#     ])
#     mask = np.zeros_like(image)
#     cv2.fillPoly(mask, polygons, 255)
#     return mask

# image = cv2.imread('../image/stop_line_1.png', 1)
# image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)

# lane_image = np.copy(image)
# canny = canny(lane_image)

# cv2.imshow("result", region_of_interest(canny))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# matplotlib Labrary 사용
# 다양한 데이터를 시각화할 수 있는 함수의 라이브러리(그래프 등)
import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # 3 Channel -> 1 Channel
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 200)
    return canny

image = cv2.imread('../image/stop_line_1.png', 1)
image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)

lane_image = np.copy(image)
canny = canny(lane_image)

plt.imshow(canny)
plt.show()
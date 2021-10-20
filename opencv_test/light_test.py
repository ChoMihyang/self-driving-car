import cv2

img = cv2.imread('../image/TestTrack_2.png')

lab1 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

lab_planes1 = cv2.split(lab1)
clahe1 = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
lab_planes1[0] = clahe1.apply(lab_planes1[0])
lab1 = cv2.merge(lab_planes1)
clahe_bgr1 = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)
gray_bgr1 = cv2.cvtColor(clahe_bgr1, cv2.COLOR_BGR2GRAY)

img = cv2.resize(img, None, fx=0.3, fy=0.3)
result = cv2.resize(gray_bgr1, None, fx=0.3, fy=0.3)

cv2.imshow("src", img)
cv2.imshow("result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
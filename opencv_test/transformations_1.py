import cv2
import numpy as np

# bird eye view Test
img = cv2.imread('../image/TestTrack_1.png')

# height : 행, width : 열 
height = img.shape[0]
width = img.shape[1]

# 이미지 축소
shrink = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

# Manual Size 지정
zoom1 = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

# 배수 Size 지정
zoom2 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# 이미지 띄우기
# cv2.imshow('Original', img)
cv2.imshow('shrink', shrink)
# cv2.imshow('zoom1', zoom1)
# cv2.imshow('zoom2', zoom2)

cv2.waitKey(0)
cv2.destroyAllWindows()

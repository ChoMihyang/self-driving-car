import cv2 
import numpy as np 
import scipy.fftpack 

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

def canny(img, canny_blur, low_thresh, high_thresh):
    lane_image = np.copy(img) 
    blur = cv2.GaussianBlur(lane_image, (5, 5), canny_blur) # GaussianBlur(?????????, ?????????, x/y?????? ????????????)
    canny = cv2.Canny(blur, low_thresh, high_thresh) # Canny(?????????, ?????? ?????????, ?????? ?????????)  

    return canny

# Video ????????????
cap = cv2.VideoCapture("../image/Shadow_Test_1.avi")

while True:
    ret, img = cap.read()

    if not ret:
        break

    # cvtColor & resize
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (350, 400))

    # Number of rows and columns
    rows = img.shape[0]
    cols = img.shape[1]

    # Remove some columns from the beginning and end
    img = img[:, 59:cols-20]

    # Number of rows and columns
    rows = img.shape[0]
    cols = img.shape[1]

    homo_sigma = 1
    canny_blur = 5
    low_thresh = 70
    high_thresh = 190

    # Homomorphic filtering
    Ihmf2, Ithresh, Iopen = homomorphic_filtering(img, homo_sigma)
    # canny filtering
    canny_result = canny(Ihmf2, canny_blur, low_thresh, high_thresh)

    # ????????? ?????????
    cv2.imshow("Canny Result", canny_result)
    cv2.imshow("Homo Result", Ihmf2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
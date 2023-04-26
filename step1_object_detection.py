import cv2
import numpy as np
import csv


# Callback function for trackbars
def nothing(x):
    pass


# This function is used to stack all the provided images
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


# Initializing video capture from default camera - 0
cap = cv2.VideoCapture(0)

# Creating Trackbar to find the mask value of the object
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 60, 180, nothing) # L-H : Lower Hue
cv2.createTrackbar("L-S", "Trackbars", 45, 255, nothing) # L-S : Lower Saturation
cv2.createTrackbar("L-V", "Trackbars", 90, 255, nothing) # L-V : Lower Value
cv2.createTrackbar("U-H", "Trackbars", 180, 180, nothing) # U-H : Upper Hue
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing) # U-S : Upper Saturation
cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing) # U-V : Upper Value

# Defining font for marking the object from video
font = cv2.FONT_HERSHEY_COMPLEX

# Keep capturing video streams until "q" button pressed
while True:
    success, frame = cap.read()

    # Break if video stream capture is failed
    if not success:
        break

    # Convert the frames to HSV color space from default BGR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Set lower and upper color bounds for the defined object from trackbar
    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])

    # Masking for detecting defined object
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)


    # Defining contours to find the edges and bounding box of the object
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Find the area of the detected object
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        # Ignore all the other detected object whose area is smaller than 50k
        if area > 500:
            # Calculate the bounding box coordinates
            x, y, width, height = cv2.boundingRect(cnt)

            # Find the center of contour where cX gives the x coordinate of the controid and cY gives the y coordinate of the centroid
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.drawContours(frame, [approx], 0, (0,0,0), 5)

            if len(approx) == 4:
                cv2.putText(frame, "Rectangle", (cX,cY), font, 1, (0,0,0))
            elif len(approx) == 3:
                cv2.putText(frame, "Triangle", (cX, cY), font, 1, (0, 0, 0))
            else:
                cv2.putText(frame, "Undefined", (cX, cY), font, 1, (0, 0, 0))

    imgStack = stackImages(0.5,([frame,mask]))

    cv2.imshow('Stack Images', imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
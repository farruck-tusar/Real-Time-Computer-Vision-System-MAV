import cv2
import numpy as np
import csv

# Initializing video capture from default camera - 0
cap = cv2.VideoCapture(0)

# Create a CSV file to write the bounding box coordinates
csv_file = open('bounding_box_coordinates.csv', 'w')
csv_writer = csv.writer(csv_file)

# Record the video stream using codec_id for MP4
fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
record_video = cv2.VideoWriter('recorded_video.mp4', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))

# Initialize frame number
frame_number = 0

# Keep capturing video streams until "q" button pressed
while True:
    success, frame = cap.read()

    # Break if video stream capture is failed
    if not success:
        break

    # Convert the frames to HSV color space from default BGR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Set lower and upper color bounds for the defined object
    lower = np.array([15, 40, 80])
    upper = np.array([180, 255, 255])

    # Masking for detecting defined object
    mask = cv2.inRange(hsv, lower, upper)

    # Removal or erosion of noises using np matrix from the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    frame_number += 1

    # Defining contours to find the edges and bounding box of the object
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Find the area of the detected object
        area = cv2.contourArea(cnt)

        # Ignore all the other detected object whose area is smaller than 500
        if area > 500:
            # Calculate the bounding box coordinates
            x, y, width, height = cv2.boundingRect(cnt)

            # Write the bounding box coordinates to the CSV file
            csv_writer.writerow([frame_number, x, y, width, height])

    # Write the frame to the recorded video
    record_video.write(frame)
    cv2.imshow('Result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
record_video.release()
csv_file.close()
cv2.destroyAllWindows()
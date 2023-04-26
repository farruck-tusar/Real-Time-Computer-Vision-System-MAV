import cv2
import csv

# Load the recorded video
cap = cv2.VideoCapture('recorded_video.mp4')  # Change the file name if using a different video file

# Open the CSV file containing bounding box coordinates
with open('bounding_box_coordinates.csv', 'r') as csvfile:  # Change the file name if using a different CSV file
    reader = csv.reader(csvfile)
    bounding_boxes = []
    for row in reader:
        # Each row in the CSV file should contain 5 values: frame number, x, y, width, height
        frame_num = int(row[0])
        x = int(row[1])
        y = int(row[2])
        width = int(row[3])
        height = int(row[4])
        bounding_boxes.append((frame_num, x, y, width, height))

frame_num = 0  # Start from the first frame

while True:
    ret, frame = cap.read()  # Capture a frame from the video

    if not ret:
        break
    print(bounding_boxes)
    # Get the bounding boxes for the current frame from the CSV file
    while bounding_boxes and bounding_boxes[0][0] == frame_num:
        # Draw the bounding box on the frame
        _, x, y, width, height = bounding_boxes.pop(0)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Video Playback', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit
        break

    frame_num += 1

# Release the resources
cap.release()
cv2.destroyAllWindows()

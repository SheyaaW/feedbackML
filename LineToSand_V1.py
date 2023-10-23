import numpy as np
import cv2
from ultralytics import YOLO
import torch

video_path = "C:/Users/tewwa/Downloads/feedback_proj/feedback_proj/VID_20231011_164753.mp4" #camera
cap = cv2.VideoCapture(video_path)

model2 = YOLO("C:/Users/tewwa/Downloads/feedback_proj/feedback_proj/last.pt")
processed_contours = []

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret:
        results_seg = model2(frame)
        seg_frame = results_seg[0].plot_mask_only()

        # Convert image to HSV color space
        hsv = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2HSV)

        # Define range of sand color in HSV format
        lower_sand = np.array([0, 20, 70])
        upper_sand = np.array([20, 255, 255])

        # Create a mask for the sand color range
        mask = cv2.inRange(hsv, lower_sand, upper_sand)

        # Apply the mask to the original image to extract only the sand color
        sand = cv2.bitwise_and(seg_frame, seg_frame, mask=mask)

        # Convert the sand color image to grayscale
        gray = cv2.cvtColor(sand, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to the grayscale image
        blur = cv2.GaussianBlur(gray,(5,5),0)

        # Apply thresholding to the blurred image
        ret, thresh_img = cv2.threshold(blur,91,255,cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours =  cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        height, width = seg_frame.shape[:2]
        bottom_center = (int(width/2), height)

        # Draw contours for only sand color

        for c in contours:
            # Check if the contour has already been processed
            if any(np.array_equal(c, pc) for pc in processed_contours):
                continue
            # Find the center of the contour
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = (cx, cy)

                # Draw a line from the bottom center to the contour center
                cv2.line(sand, bottom_center, center, (255, 0, 0), 3)

            # Draw the contour on the sand image
            cv2.drawContours(sand, [c], -1, (0,255,0), 3)
            processed_contours.append(c)
            
        cv2.imshow('seg_frame',sand)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    # Display the resulting seg_frame
    #print(len(contours))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
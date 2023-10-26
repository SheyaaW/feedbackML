import numpy as np
import cv2
from ultralytics import YOLO
from collections import defaultdict
import processing

# Open the video file
video_path = "C:/Users/tewwa/Downloads/feedback_proj/feedback_proj/VID_20231011_164753.mp4" #camera
cap = cv2.VideoCapture(video_path)

model1 = YOLO("C:/Users/tewwa/Downloads/feedback_proj/feedback_proj/detec.pt")
model2 = YOLO("C:/Users/tewwa/Downloads/feedback_proj/feedback_proj/last.pt")
processed_contours = []
drawPoints = []

track_history = defaultdict(lambda: [])

# Loop through the frames of the video
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if ret:
        results_det = model1.track(frame, persist=True)
        results_seg = model2(frame)
        object_mask_np = results_seg[0].plot_mask_only()
        seg_frame = results_seg[0].plot()
        annotated_frame = results_det[0].plot()
        
        hsv = cv2.cvtColor(object_mask_np, cv2.COLOR_BGR2HSV)
        # Define the color range to check (in BGR format)
        lower_sand = np.array ([10, 100, 20])
        upper_sand = np.array ([30, 255, 200])
        
        # Create a mask for the sand color range
        mask = cv2.inRange(hsv, lower_sand, upper_sand)

        # Apply the mask to the original image to extract only the sand color
        sand = cv2.bitwise_and(object_mask_np, object_mask_np, mask=mask)

        # Convert the sand color image to grayscale
        gray = cv2.cvtColor(sand, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to the grayscale image
        blur = cv2.GaussianBlur(gray,(5,5),0)

        # Apply thresholding to the blurred image
        thresh_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # Find contours in the thresholded image
        contours =  cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
        
        #annotated_frame = results_det[0].plot()

        # Draw the contours on the frame
        try:
            boxes = results_det[0].boxes.xywh.cpu()
            #print(results_det[0].boxes)
            track_ids = results_det[0].boxes.id.int().cpu().tolist()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(sand, [points], isClosed=False, color=(230, 230, 230), thickness=3)
                # ------------------Connect the lines from head to tail-----------------------
                # Draw a line between the last point of the previous track and the first point of the current track
                if len(track_history[track_id]) > 1:  
                    prev_track = track_history[track_id-1]
                    if len(prev_track) > 0:
                        prev_point = prev_track[-1]
                        curr_point = track[0]
                        cv2.line(sand, (int(prev_point[0]), int(prev_point[1])), (int(curr_point[0]), int(curr_point[1])), (230, 230, 230), thickness=3)
            
            cv2.drawContours(sand, contours, -1, (0, 255, 0), 3)
        except:
            continue

        # Display the resulting frame
        cv2.imshow('object_mask_np', sand)
        cv2.imshow('seg', sand)
        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and writer objects, and close all windows
cap.release()
cv2.destroyAllWindows()
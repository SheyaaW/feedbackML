from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO
import torch

# Load the YOLOv8 model
model1 = YOLO("C:/Users/tewwa/Downloads/feedback_proj/feedback_proj/detec.pt")
model2 = YOLO("C:/Users/tewwa/Downloads/feedback_proj/feedback_proj/last.pt")

# Open the video file
video_path = "C:/Users/tewwa/Downloads/feedback_proj/feedback_proj/VID_20231011_164753.mp4" #camera
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results_det = model1.track(frame, persist=True)
        results_seg = model2(frame)


        # Get the boxes and track IDs
        # Visualize the results on the frame
        object_mask_np = results_seg[0].plot_mask_only()
        annotated_frame = results_det[0].plot()
        seg_frame = results_seg[0].plot()
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
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        except:
            pass


        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        #cv2.imshow("YOLOv8 segm", seg_frame)
        cv2.imshow("mask segm", object_mask_np)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()





from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from utils import extrack_mask, optimal_leaf, coordinate_change, get_leaf_coor, get_control
import argparse
from time import sleep
import keyboard
import time
import subprocess
import socket
from random import randint

# Raspberry Pi's IP address and port (make sure they match the server)

host = '192.168.222.78'  # Replace with Raspberry Pi's actual IP address
port = 4444

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((host, port))



# Load the YOLOv8 model
model1 = None
model2 = None
keep_track = False
best_track_id = 0
n_try = 0
position_dict = {-5 : "1", -4 : "2", -3 : "3", -2 : "4", -1 : "5", 0 : "6", 1 : "7", 2 : "8", 3 : "9", 4 : "10", 5 : "-"}

# Store the track history
track_history = defaultdict(lambda: [])


def track_det(frame, x_shape, y_shape):
    global model1
    global model2
    global keep_track
    global n_try
    global track_history
    global best_track_id

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results_det = model1.track(frame, persist=True)
    results_seg = model2(frame)
    # Visualize the results on the frame
    object_mask_np = extrack_mask(results_seg[0].plot_mask_only())
    leaf_x, leaf_y, annotated_frame, keep_track, best_track_id, n_try = get_leaf_coor(results_det, track_history, keep_track, best_track_id, n_try, x_shape, y_shape)
    
    return annotated_frame, object_mask_np, leaf_x, leaf_y

def Dodge(frame):
    # results_seg = model2(frame)
    results_seg = model3(frame)
    object_mask_np = results_seg[0].plot_mask_only()
    processed_contours = []
    decision = "nothing"
    #Define range of grass color in HSV format
    if object_mask_np.shape[-1] == 1:
        object_mask_np = cv2.cvtColor(object_mask_np, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(object_mask_np, cv2.COLOR_BGR2HSV)
    # Define range of grass color in HSV format
    lower_people = np.array([0, 50, 50])
    upper_people = np.array([0, 210, 210])

    # Create a mask for the people color range
    mask = cv2.inRange(hsv, lower_people, upper_people)

    # Apply the mask to the original image to extract only the people color
    people = cv2.bitwise_and(object_mask_np, object_mask_np, mask=mask)

    # Convert the people color image to grayscale
    gray = cv2.cvtColor(people, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # Apply adaptive thresholding to the blurred image
    thresh_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours in the thresholded image
    contours =  cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Find the bottom center point of the frame
    height, width = object_mask_np.shape[:2]
    bottom_center = (int(width/2), height)

    # Find the largest contour
    largest_contour = None
    largest_contour_area = 0
    for c in contours:
        # Check if the contour has already been processed
        if any(np.array_equal(c, pc) for pc in processed_contours):
            continue

        # Find the area of the contour
        area = cv2.contourArea(c)

        # Update the largest contour if necessary
        if area > largest_contour_area:
            largest_contour = c
            largest_contour_area = area

    overlap_l = False
    overlap_r = False
    for i in range(50):
        if people[int(480*0.8)][320+i][2] in range(50,255):
            overlap_r = True
            break
        if people[int(480*0.8)][320-i][2] in range(50,255):
            overlap_l = True
            break

    cv2.line(people, (0, int(480*0.8)), (640, int(480*0.8)),(255, 0, 0), 3)

    # Draw a line from the bottom center to the center of the largest contour
    if largest_contour is not None:
        cv2.drawContours(people, [largest_contour], -1, (0,255,0), 3)

        # Add the largest contour to the list of processed contours
        processed_contours.append(largest_contour)

    # Determine whether to turn left or right based on the overlap
    if not overlap_r and not overlap_l:
        cv2.putText(people, "nothing!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        if overlap_l and overlap_r:
            decision = "lucky"
            cv2.putText(people, "lucky", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if overlap_l:
            decision = "left"
            cv2.putText(people, "turn left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif overlap_r:
            decision = "right"
            cv2.putText(people, "turn right", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    return decision, people

def StopAtGrass(frame):
    results_seg = model2(frame)
    object_mask_np = results_seg[0].plot_mask_only()
    processed_contours = []
    #Define range of grass color in HSV format
    if object_mask_np.shape[-1] == 1:
        object_mask_np = cv2.cvtColor(object_mask_np, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(object_mask_np, cv2.COLOR_BGR2HSV)
    # Define range of grass color in HSV format
    lower_grass = np.array([0, 50, 50])
    upper_grass = np.array([0, 210, 210])

    # Create a mask for the grass color range
    mask = cv2.inRange(hsv, lower_grass, upper_grass)

    # Apply the mask to the original image to extract only the grass color
    grass = cv2.bitwise_and(object_mask_np, object_mask_np, mask=mask)

    # Convert the grass color image to grayscale
    gray = cv2.cvtColor(grass, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # Apply adaptive thresholding to the blurred image
    thresh_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours in the thresholded image
    contours =  cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Find the bottom center point of the frame
    height, width = object_mask_np.shape[:2]
    bottom_center = (int(width/2), height)

    # Find the largest contour
    largest_contour = None
    largest_contour_area = 0
    for c in contours:
        # Check if the contour has already been processed
        if any(np.array_equal(c, pc) for pc in processed_contours):
            continue

        # Find the area of the contour
        area = cv2.contourArea(c)

        # Update the largest contour if necessary
        if area > largest_contour_area:
            largest_contour = c
            largest_contour_area = area

    # Draw a line from the bottom center to the center of the largest contour
    if largest_contour is not None:
        # Find the center of the largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
        # Draw a line from the bottom center to the contour center
            cv2.line(grass, bottom_center, center, (255, 0, 0), 3)
        # Draw the largest contour on the grass image
        cv2.drawContours(grass, [largest_contour], -1, (0,255,0), 3)

        # Add the largest contour to the list of processed contours
        processed_contours.append(largest_contour)

    return grass, center

#-------------------------------------main--------------------------------------------#
def main(opt):
    # Loop through the video frames
    global model1
    global model2
    global model3
    model1 = YOLO(opt.det_path)
    model2 = YOLO(opt.seg_path)
    model3 = YOLO(opt.follow)

    # Open the video file
    video_path = opt.video #camera
    try:
        video_path = int(video_path)
    except:
        pass

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            annotated_frame, object_mask_np, leaf_x, leaf_y = track_det(frame, opt.x_shape, opt.y_shape)
            color_frame = annotated_frame.copy()
            sliced_x, sliced_y = get_control(leaf_x, leaf_y, opt.x_shape, opt.y_shape)
            GrassStop, GrassCenter = StopAtGrass(frame)
            comm, dodge_people = Dodge(frame)
            
            if opt.show_img:
                # cv2.imshow("YOLOv8 Tracking", annotated_frame)
                #cv2.imshow("Grass Segmentation", GrassStop)
                cv2.imshow("following", dodge_people)
                cv2.imshow("",frame)
                print(f"Grass Center ({GrassCenter}")
                print(f"leaf ({leaf_x}, {leaf_y})")
                print(f"sliced ({sliced_x}, {sliced_y})")
                key = cv2.waitKey(1)
                if key == ord("q"):
                    # Release the keyboard hook when the program exits.
                    keyboard.unhook_all()
                    client_socket.close()
                    break
                elif key == ord("i"):
                    if comm == "left":
                        action = "1\n"
                        client_socket.send(comm.encode('utf-8'))
                    elif comm == "right":
                        action = "-\n"
                        client_socket.send(comm.encode('utf-8'))
                    elif comm == "lucky":
                        idx = randint(0, 1)
                        if idx == 0:
                            action = "1\n"
                        else: 
                            action = "-\n"
                    else:
                        action = position_dict[sliced_x] + "\n"
                    
                    client_socket.send(action.encode('utf-8'))
                    
                elif key == ord("f"):
                    action = "f\n"
                    client_socket.send(action.encode('utf-8'))
                    
                elif key == ord("t"):
                    action = "t\n"
                    client_socket.send(action.encode('utf-8'))
                    
                elif key == ord("u"):
                    action = "u\n"
                    client_socket.send(action.encode('utf-8'))
                    
                else:
                    action = "p\n"
                    client_socket.send(action.encode('utf-8'))
                    
                print(f"leaf ({leaf_x}, {leaf_y})")
                
                cv2.putText(annotated_frame, f"leaf ({leaf_x}, {leaf_y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("mask segm", object_mask_np)
                cv2.imshow("segm", annotated_frame)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    # Release the keyboard hook when the program exits.
                    keyboard.unhook_all()
                    client_socket.close()
                    break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    if opt.show_img:
        cap.release()
        cv2.destroyAllWindows()
#-------------------------------------main--------------------------------------------#

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_path', type=str, help='path of detection model', default=r"C:\Users\tewwa\FeedbackSegment\fb\best.pt", required=False)
    parser.add_argument('--seg_path', type=str, help='path of segmentation model', default=r"C:\Users\tewwa\FeedbackSegment\fb\last.pt", required=False)
    parser.add_argument('--follow', type=str, help='path of segmentation model', default="yolov8n-seg.pt", required=False)
    # parser.add_argument('--video', type=int, help='path of video or video device or http', default=0)
    parser.add_argument('--video', type=str, help='path of video or video device or http', default="https://192.168.222.145:8080/video", required=False)
    parser.add_argument('--show_img', type=int, help='if show image set 1 else 0', default=1, required=False)
    parser.add_argument('--x_shape', type=int, help='screen range in x axis', default=640, required=False)
    parser.add_argument('--y_shape', type=int, help='screen range in y axis', default=480, required=False)
    #parser.add_argument()
    opt = parser.parse_args()

    main(opt)



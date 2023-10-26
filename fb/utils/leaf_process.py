from utils import optimal_leaf
import cv2
import numpy as np

def get_leaf_coor(results_det, track_history, keep_track, best_track_id, n_try,  x_shape, y_shape):
    
    annotated_frame = results_det[0].plot()

    try:
        boxes = results_det[0].boxes.xywh.cpu()
        track_ids = results_det[0].boxes.id.int().cpu().tolist()

        # Plot the tracks
        x_s = []
        y_s = []
        tracks_list = []
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            x_s.append(float(x))
            y_s.append(float(y))
            tracks_list.append(track_id)
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)
            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        

        x_s, y_s = np.array(x_s), np.array(y_s)

        if keep_track==False:
            xst, yst, best_idx = optimal_leaf(x_s, y_s)
            best_track_id = tracks_list[best_idx]
            #print(f'vaslues...........{best_track_id}')
            keep_track = True
            color = (0, 0, 255)
            thickness = 1
            cv2.line(annotated_frame, (int(x_shape/2),int(y_shape)), (int(xst),int(yst)), color, thickness) #edit resolution here!!!!

            #ID extracktor
            #or do other thing
            #consistant tracking
        
        else:
            #find what is the idx of best_track_id
            #if delete tracking set keep_track to False
            if best_track_id in tracks_list:
                n_try = 0
                wanted_idx = tracks_list.index(best_track_id)
                color = (0, 0, 255)
                thickness = 1
                cv2.line(annotated_frame, (int(x_shape/2),int(y_shape)), (int(x_s[wanted_idx]),int(y_s[wanted_idx])), color, thickness)
                print('keep tracking!!!!!!!!!!!!!!!!!')
                return x_s[wanted_idx], y_s[wanted_idx], annotated_frame, keep_track, best_track_id, n_try
            else:
                n_try += 1
                #print('222222@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                #print(n_try)
                color = (0, 0, 255)
                thickness = 1
                cv2.line(annotated_frame, (int(1920/2),int(1080)), ( int(1920/2),int(1080/2)), color, thickness)
                if n_try == 30:
                    #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                    #already got the leaf
                    keep_track = False

    except:
        pass

    return int(x_shape/2),int(y_shape/2), annotated_frame, keep_track, best_track_id, n_try
    
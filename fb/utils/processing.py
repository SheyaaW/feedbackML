import cv2
import numpy as np

def postprocess(final_mask):
    image = cv2.cvtColor(final_mask, cv2.COLOR_RGB2BGR)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    eroded = cv2.dilate(image, kernel, iterations=20)
    #to binary image
    threshold_value = 150  # Set the threshold value
    max_value = 255  # Maximum value assigned to pixels above the threshold
    _, binary_image = cv2.threshold(eroded, threshold_value, max_value, cv2.THRESH_BINARY)
    # Set white and black pixels
    binary_image[binary_image == 255] = 255
    binary_image[binary_image == 0] = 0
    # Load the segmentation mask as a grayscale image
    segmentation_mask = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    return segmentation_mask

    

def extrack_mask(object_mask_np):
    '''brown color in this case'''
    hsv = cv2.cvtColor (object_mask_np, cv2.COLOR_BGR2HSV)
    lower_brown = np.array ([10, 100, 20]) 
    upper_brown = np.array ([30, 255, 200])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    fn_mask = postprocess(mask)
    return fn_mask



def optimal_leaf(x, y):
    best_x = x[0]
    best_y = y[0]
    best_rsme = 1e6
    idx = 0
    for i in range(len(x)):
        # Choose the data point where the line ends
        x_end = x[i]
        y_end = y[i]

        # Calculate the slope for the line through (0,0) and (x_end, y_end)
        m_end = y_end / x_end

        # Generate the line through (0,0) and (x_end, y_end)
        y_fit = m_end * x_end

        # Calculate RMSE
        squared_errors = (y - y_fit) ** 2
        rmse = np.sqrt(squared_errors.mean())
        #print("rms", rmse)
        if rmse < best_rsme:
            best_rsme = rmse
            best_x = x_end
            best_y = y_end
            idx = i

    return best_x, best_y, idx

def coordinate_change(x,y, a=360,b=640):
    return x-a,b-y


def get_control(x,y, x_shape,y_shape):
    x_zone = x // (x_shape // 11)
    y_zone = y // (y_shape // 11)
    x_zone = np.clip(x_zone, a_min = -5, a_max = 5) 
    y_zone = np.clip(y_zone, a_min = -5, a_max = 5) 
    return x_zone, y_zone
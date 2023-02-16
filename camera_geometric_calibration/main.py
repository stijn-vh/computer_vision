import online_phase as Online
import offline_phase as Offline
import numpy as np
import cv2 as cv

# Configure global variables and call necessary offline/online functions
if __name__ == '__main__':
    num_cols = 9
    num_rows = 6
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    image_name = 'current image'
    objp = np.zeros((num_cols * num_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_cols, 0:num_rows].T.reshape(-1, 2)
    
    config = {
        'criteria': criteria, 
        'num_cols': num_cols, 
        'num_rows': num_rows, 
        'image_name': image_name,
        'objp': objp
    }

    Offline.set_config(config)
    Online.set_config(config)

    params = Offline.execute_offline_phase()
    Online.execute_online_phase(params)

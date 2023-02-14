import online_phase as Online
import offline_phase as Offline
import numpy as np
import cv2 as cv

if __name__ == '__main__':
    num_cols = 9
    num_rows = 6
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    config = {'criteria': criteria, 'num_cols': num_cols, 'num_rows': num_rows}
    Offline.set_config(config)
    params = Offline.execute_offline_phase()
    print('Offline executed')
    Online.set_config(config)
    Online.execute_online_phase(params)

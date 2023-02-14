import numpy as np
import cv2 as cv

def direction_step(p1, p2, num_steps):
    x1, y1 = p1
    x2, y2 = p2
    return (x2 - x1) / num_steps, (y2 - y1) / num_steps

def interpolate_three_corners(three_corners):
    # traverse rows towards the middle point, traverse cols away from middle point
    upper_right_cor, lower_right_cor, lower_left_cor = three_corners
    ur_corx, ur_cory = upper_right_cor

    # direction is also scaled to stepsize.
    row_dirx, row_diry = direction_step(upper_right_cor, lower_right_cor, (num_rows - 1))
    col_dirx, col_diry = direction_step(lower_right_cor, lower_left_cor, (num_cols - 1))

    index = 0
    corners = np.zeros((num_rows * num_cols, 1, 2))  # same shape as the findChessboardCorners output
    for j in range(num_rows):
        currentx = ur_corx + j * row_dirx
        currenty = ur_cory + j * row_diry
        for i in range(num_cols):
            corners[index] = [currentx, currenty]
            index += 1
            currentx += col_dirx
            currenty += col_diry
    return corners

"""
num_cols = 5
num_rows = 3
x = interpolate_three_corners([(5, 1), (5, 3), (1, 3)])
print("interpolation of [(5, 1), (5, 3), (1, 3)] = \n", x)

num_cols = 3
num_rows = 3
x = interpolate_three_corners([(2, 0), (4, 2), (2, 4)])
print("interpolation of [(2, 0), (4, 2), (2, 4)] = \n", x)
"""



num_cols = 9
num_rows = 6

img = cv.imread("images/WIN_20230212_15_51_43_Pro.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (num_cols, num_rows), None)

corners2 = interpolate_three_corners([(366, 237), (253, 333), (127, 222)])
corners2 = np.float32(corners2)
print("interpolation of [(366, 237), (253, 333), (127, 222)] = \n", corners2)

cv.drawChessboardCorners(img, (num_cols, num_rows), corners2, ret)
cv.imshow('current_image', img)
cv.waitKey(10000)
cv.destroyAllWindows()
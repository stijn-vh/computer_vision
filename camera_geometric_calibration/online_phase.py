
# From slides: 
# Workflow online:
# • Read an image/camera frame
# • Draw a box on a detected chessboard in the right perspective

class online_phase:
    test_image = None

    # take the test image and draw the world 3D axes (XYZ) with the origin at the center 
    # of the world coordinates, using the estimated camera parameters
    def draw_axes_on_image(image, estimated_camera_params):
        return

    # draw a cube which is located at the origin of the world coordinates. 
    # You can get bonus points for doing this in real time using your webcam
    def draw_cube_on_image(image, real_time = False):
        return

    def execute_online_phase(estimated_camera_params):
        return
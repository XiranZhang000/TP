import cv2
import numpy as np
import sys

cl_points = []

# Function called when the user clicks a point in an image while determining scale
def mouse_callback_right(event, x, y, flags, image):
    if len(cl_points) == 2:
        return
    elif event == cv2.EVENT_LBUTTONDOWN:
        print("Pointed clicked at x, y = ", x, y)
        cl_points.append([x, y])
        cv2.circle(image, (x, y), 3, (0, 0, 0), -1)
        cv2.imshow("TP image", image)

if __name__ == '__main__':
    print("Program use is as follows: python tp3-2021-undistortion.py distorted_image camera_parameters_xml_file")
    if len(sys.argv) != 3:
       sys.exit("Did not provide correct number of arguments, will exit.")  

    s = cv2.FileStorage()  # Projection matrix obtained by calibrating your camera

    image_to_open = sys.argv[1] # Image to be used
    cam_param_file = sys.argv[2] # XML file containing calibration parameters calculated from OpenCV

    s.open(cam_param_file, cv2.FileStorage_READ) # Retrieve camera parameters from saved calibration data

    P = np.float32(s.getNode('cameraMatrix').mat()) # Retrieve camera matrix
    dist = np.float32(s.getNode('dist_coeffs').mat()) # Retrieve distortion coefficients    
    #######################################################

    # 1.2 Undistort radial lens distortion
    line_color = (255, 255, 255)
    img = cv2.imread(image_to_open) # Read the distorted image provided as first argument
    cv2.namedWindow("TP image")
    cv2.setMouseCallback("TP image", mouse_callback_right, img)
    cv2.imshow("TP image", img)
    print("Click on a pair of points that are collinear in the real world. Finally, press enter within the OpenCV window to continue.")
    key = cv2.waitKey(0) # Waiting to press enter in the shown image to continue

    cv2.line(img, tuple(cl_points[0]), tuple(cl_points[1]), line_color, 1, cv2.LINE_AA)
    cv2.imshow("TP image", img)
    print("Line connecting the two points is shown. Press enter (within the OpenCV window) to continue")
    key = cv2.waitKey(0) # Waiting to press enter in the shown image to continue
    cv2.imwrite('distorted-with-distorted-point-pair.jpg', img)

    img = cv2.imread(image_to_open)
    undistorted_image = cv2.undistort(img, P[:3, :3], dist, P)
    cv2.namedWindow("TP image")

    cl_points2 = cv2.undistortPoints(np.float32(cl_points), P[:3, :3], dist, np.eye(3), P) # cf. https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html#ga55c716492470bfe86b0ee9bf3a1f0f7e

    cv2.circle(undistorted_image, tuple(int(cl_points2[0][0])), 3, (0, 0, 0), -1)
    cv2.circle(undistorted_image, tuple(int(cl_points2[1][0])), 3, (0, 0, 0), -1)
    cv2.line(undistorted_image, tuple(int(cl_points2[0][0])), tuple(cl_points2[1][0]), line_color, 1, cv2.LINE_AA)
    cv2.imshow("TP image", undistorted_image)
    print("Original line shown in undistorted image. Press enter (within the OpenCV window) to exit")
    cv2.imwrite('undistorted-with-undistorted-point-pair.jpg', undistorted_image)
    key = cv2.waitKey(0) # Waiting to press enter in the shown image to continue
    #######################################################

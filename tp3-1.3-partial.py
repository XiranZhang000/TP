import cv2
import numpy as np
import sys
import os

cube_vertices = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1], [0, 0.5, -1.4], [1, 0.5, -1.4]])
cube_vertices *= 0.1 # reducing the scale of the cube-house

cube_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 8), (6, 9), (7, 9), (8, 9)] # Every pair contains the indices of the vertices that are connected by an edge

if __name__ == '__main__':
    print("Program use is as follows: python3 tp3-1.3-partial.py <number_of_corners_horizontally> <number_of_corners_vertically> <distance_between_corners_in_meters> <camera_id> <path_to_video>(Optional). E.g. python3 tp4-1.3-partial.py 9 6 0.025 0 'video-to-apply-ar.mp4'")
    if len(sys.argv)==6:
        cap = cv2.VideoCapture(sys.argv[5])
        width = int(sys.argv[1]) # number of corners horizontally
        height = int(sys.argv[2]) # number of corners vertically
        edgesize = float(sys.argv[3]) # length of square size, should be in metters
        camera_id = int(sys.argv[4]) # The id of the camera to open in case multiple cameras are connected. Its value should be 0 if there is only one camera
        chesscorners = np.zeros((width*height, 3), np.float32)
        for i in range(0, width):
            for j in range(0, height):
                chesscorners[i*height + j] = [j, i, 0]
        chesscorners *= edgesize

        #cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION) # Retrieve frames using camera

        if not cap.isOpened():
            print("Cannot open video capture")
            exit()
        # Retrieve saved extrinstic and intrinsic camera parameters and assign
        s = cv2.FileStorage()
        os.chdir(sys.path[0])
        s.open('data/teacher-camera/cameraParameters.xml', cv2.FileStorage_READ) # Copy your calibration parameter file in the specified folder

        # YOUR CODE: Set /Users/xiran/Downloads/TP3-material 4/data/teacher-camera/cameraParameters.xmlvideo width capture resolution to correspond to the width calibration resolution
        res = s.getNode('cameraResolution')
        ou_dir = 'data/tp3-video-applied.mp4'
        fps = cap.get(cv2.CAP_PROP_FPS)
        videowriter = cv2.VideoWriter(ou_dir, cv2.VideoWriter_fourcc(*'mp4v'),fps, (int(res.at(0).real()),int(res.at(1).real())))
        #cap.set(3, res.at(0).real())
        #res.at(0).real()   #calibration resolution
        #cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH)     #video width capture resolution
        # YOUR CODE: Set video height capture resolution to correspond to the height calibration resolution
        #cap.set(4, res.at(1).real())
        # YOUR CODE: Retrieve camera matrix
        P = np.float32(s.getNode('cameraMatrix').mat())
        # YOUR CODE: Retrieve distortion coefficients
        dist = np.float32(s.getNode('dist_coeffs').mat())
        while True:
            print('Detecting the chessboard...')
            ret,frame = cap.read() # Capture frame-by-frame. If frame is read correctly ret is True
            if ret:
                frame = cv2.resize(frame, (int(res.at(0).real()),int(res.at(1).real())), cv2.INTER_LINEAR)
                # YOUR CODE: Undistort every frame using known calibration parameters
                undistorted_frame = cv2.undistort(frame, P[:3, :3], dist, P)
                # YOUR CODE: Look for the chessboardcorners in the undistorted frame
                retval, corners = cv2.findChessboardCorners(undistorted_frame, (height,width),None)
                # YOUR CODE: Recover the rotation and translation vectors by estimating the transformation between projected chessboard and original chessboard
                if retval:
                    retval2,R,t = cv2.solvePnP(chesscorners,corners,P,dist)
                    # YOUR CODE: Project cube-house vertices to camera image using the known extracted extrinsic parameters
                    if retval2:
                        cube_vertices_c,_ = cv2.projectPoints(cube_vertices,R,t,P,dist)
                        # YOUR CODE: Draw a line for every pair of vertices that is connected by an edge
                        line_color = (255, 255, 255)
                        try:
                            for i in range(len(cube_edges)):
                                undistorted_frame = cv2.line(undistorted_frame,tuple(map(int,cube_vertices_c[cube_edges[i][0]].tolist()[0])),tuple(map(int,cube_vertices_c[cube_edges[i][1]].tolist()[0])),color = line_color)
                        except:
                            pass
                cv2.imshow('tp',undistorted_frame)
                cv2.waitKey(1)
                videowriter.write(undistorted_frame)
            if not ret:
                break
        # YOUR CODE: Save frame to video file
        cap.release() # When everything done, release the cachoose plusieurs colonnespture
        cv2.destroyAllWindows()

    elif len(sys.argv)==5:
        print('Press q to exit')
        width = int(sys.argv[1]) # number of corners horizontally
        height = int(sys.argv[2]) # number of corners vertically
        edgesize = float(sys.argv[3]) # length of square size, should be in metters
        camera_id = int(sys.argv[4]) # The id of the camera to open in case multiple cameras are connected. Its value should be 0 if there is only one camera
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L) # Retrieve frames
        # chesscorners contains the coordinates of a width x height array of 3D points lying on a XY plane with Z coordinate equal to 0. It corresponds to the real-world expected coordinates of corner points on the used chessboard
        chesscorners = np.zeros((width*height, 3), np.float32)
  
        for i in range(0, width):
          for j in range(0, height):
              chesscorners[i*height + j] = [j, i, 0]
        chesscorners *= edgesize

        #cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION) # Retrieve frames using camera

        if not cap.isOpened():
          print("Cannot open video capture")
          exit()
        # Retrieve saved extrinstic and intrinsic camera parameters and assign
        s = cv2.FileStorage()
        os.chdir(sys.path[0])
        s.open('data/teacher-camera/cameraParameters.xml', cv2.FileStorage_READ) # Copy your calibration parameter file in the specified folder
        # YOUR CODE: Set video width capture resolution to correspond to the width calibration resolution
        res = s.getNode('cameraResolution')
        fps = cap.get(cv2.CAP_PROP_FPS)
        ou_dir = 'data/tp3-video-applied.mp4'
        videowriter = cv2.VideoWriter(ou_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps*6, (int(res.at(0).real()),int(res.at(1).real())))
        cap.set(3, int(res.at(0).real()))
        #res.at(0).real()   #calibration resolution
        #cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH)     #video width capture resolution
        # YOUR CODE: Set video height capture resolution to correspond to the height calibration resolution
        cap.set(4, int(res.at(1).real()))
        # YOUR CODE: Retrieve camera matrix
        P = np.float32(s.getNode('cameraMatrix').mat())
        # YOUR CODE: Retrieve distortion coefficients
        dist = np.float32(s.getNode('dist_coeffs').mat())
        while True:
            ret,frame = cap.read() # Capture frame-by-frame. If frame is read correctly ret is True
            if ret:
              # YOUR CODE: Undistort every frame using known calibration parameters
              undistorted_frame = cv2.undistort(frame, P[:3, :3], dist, P)
              # YOUR CODE: Look for the chessboardcorners in the undistorted frame
              retval, corners = cv2.findChessboardCorners(undistorted_frame, (height,width),None)
              # YOUR CODE: Recover the rotation and translation vectors by estimating the transformation between projected chessboard and original chessboard
              if retval:
                    retval2,R,t = cv2.solvePnP(chesscorners,corners,P,dist)
                    # YOUR CODE: Project cube-house vertices to camera image using the known extracted extrinsic parameters
                    if retval2:
                        cube_vertices_c,_ = cv2.projectPoints(cube_vertices,R,t,P,dist)
                        # YOUR CODE: Draw a line for every pair of vertices that is connected by an edge
                        line_color = (255, 255, 255)
                        try:
                            for i in range(len(cube_edges)):
                                undistorted_frame = cv2.line(undistorted_frame,tuple(map(int,cube_vertices_c[cube_edges[i][0]].tolist()[0])),tuple(map(int,cube_vertices_c[cube_edges[i][1]].tolist()[0])),color = line_color)
                        except:
                            pass
              cv2.imshow('tp',undistorted_frame)
              cv2.waitKey(1)
              videowriter.write(undistorted_frame)
            if cv2.waitKey(1) == ord('q'):
                break
            # YOUR CODE: Save frame to video file
        cap.release() # When everything done, release the cachoose plusieurs colonnespture
        cv2.destroyAllWindows()
    else:
        os.chdir(sys.path[0])
        cap = cv2.VideoCapture('data/tp3-video-to-apply-ar.mp4')
        width = 9
        height = 6
        edgesize = 0.025
        chesscorners = np.zeros((width*height, 3), np.float32)
        for i in range(0, width):
            for j in range(0, height):
                chesscorners[i*height + j] = [j, i, 0]
        chesscorners *= edgesize
        if not cap.isOpened():
            print("Cannot open video capture")
            exit()
        s = cv2.FileStorage()
        os.chdir(sys.path[0])
        s.open('data/teacher-camera/cameraParameters.xml', cv2.FileStorage_READ)
        res = s.getNode('cameraResolution')
        fps = cap.get(cv2.CAP_PROP_FPS)
        ou_dir = 'data/tp3-video-applied.mp4'
        videowriter = cv2.VideoWriter(ou_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(res.at(0).real()),int(res.at(1).real())))
        P = np.float32(s.getNode('cameraMatrix').mat())
        dist = np.float32(s.getNode('dist_coeffs').mat())
        while True:
            print('Detecting the chessboard...')
            ret,frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (int(res.at(0).real()),int(res.at(1).real())), cv2.INTER_LINEAR)
                undistorted_frame = cv2.undistort(frame, P[:3, :3], dist, P)
                retval, corners = cv2.findChessboardCorners(undistorted_frame, (height,width),None)
                # YOUR CODE: Recover the rotation and translation vectors by estimating the transformation between projected chessboard and original chessboard
                if retval:
                    retval2,R,t = cv2.solvePnP(chesscorners,corners,P,dist)
                    # YOUR CODE: Project cube-house vertices to camera image using the known extracted extrinsic parameters
                    if retval2:
                        cube_vertices_c,_ = cv2.projectPoints(cube_vertices,R,t,P,dist)
                        # YOUR CODE: Draw a line for every pair of vertices that is connected by an edge
                        line_color = (255, 255, 255)
                        try:
                            for i in range(len(cube_edges)):
                                undistorted_frame = cv2.line(undistorted_frame,tuple(map(int,cube_vertices_c[cube_edges[i][0]].tolist()[0])),tuple(map(int,cube_vertices_c[cube_edges[i][1]].tolist()[0])),color = line_color)
                        except:
                            pass
                cv2.imshow('tp',undistorted_frame)
                cv2.waitKey(1)
                videowriter.write(undistorted_frame)
            if not ret:
                break
        # YOUR CODE: Save frame to video file
        cap.release() # When everything done, release the cachoose plusieurs colonnespture
        cv2.destroyAllWindows()


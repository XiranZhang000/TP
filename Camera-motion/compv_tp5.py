import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy
from scipy.spatial.transform import Rotation
import sys

if __name__ == '__main__':
    
  root = sys.argv[1] #the path to the camera calibration file
  img1_path = sys.argv[2] #path to the first image
  img2_path = sys.argv[3]#path to the second image
  s = cv.FileStorage()
  s.open(root, cv.FileStorage_READ)
  P= np.float32(s.getNode('cameraMatrix').mat())
  dist = np.float32(s.getNode('dist_coeffs').mat())
  img1 = cv.imread(img1_path)
  img2 = cv.imread(img2_path)
  # Undistort the images
  img1 = cv.undistort(img1, P[:3, :3], dist, P)
  img2 = cv.undistort(img2, P[:3, :3], dist, P)
  # Find the matches between two images
  gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
  sift = cv.xfeatures2d.SIFT_create()
  gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
  kp1, des1 = sift.detectAndCompute(gray1,None)
  kp2, des2 = sift.detectAndCompute(gray2,None)
  bf = cv.BFMatcher()
  matches = bf.knnMatch(des1,des2,k=2)
  good = []
  for m,n in matches:
    if m.distance < 0.55*n.distance:
          good.append([m])
  good = sorted(good, key = lambda x:x[0].distance)
  points1,points2 = [],[]
  for i in range(len(good)):
    points1.append(np.array(kp1[good[i][0].queryIdx].pt))
    points2.append(np.array(kp2[good[i][0].trainIdx].pt))
  points1 = np.array(points1)
  points2 = np.array(points2)
  # Get the Fundamental matrix
  F, mask = cv.findFundamentalMat(points1,points2)
  # Get the Essential matrix
  E = np.dot(np.dot(np.transpose(P),F),P)
  # Obtain the motion of the second camera
  retval, R, t, mask = cv.recoverPose(E,points1,points2,P)
  T1 = np.transpose(R)
  T2 = -R.transpose().dot(t)
  T = np.hstack((T1,T2))
  T = np.vstack((T,np.array((0,0,0,1))))
  print('The transpose matrix is:')
  print(T)
  r = Rotation.from_matrix(R.transpose()).as_euler('xyz',degrees=True)
  print('The matrix orresponds to Euler angles:')
  print(r)
  camera1 = np.float32([[0, 0, 0, 1], [0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1]])
  camera2 = T.dot(camera1.transpose()).transpose()
 # camera2 = np.dot(camera1,T)
  print(camera2)
  # ax = plt.axes(projection='3d')
  ax = plt.figure().add_subplot(projection='3d')
  cube_edges = [(0,1),(0,2),(0,3)]
  ax.set_zlim(-1.5, 1.5)
  ax.set_xlim(-1.5, 1.5)
  ax.set_ylim(-1.5, 1.5)
  ax.set_ylabel('y')
  ax.set_xlabel('x')
  ax.set_zlabel('z')
  color = ['r','g','b']
  print(camera2[cube_edges[0][1]].tolist()[0],camera2[cube_edges[0][1]].tolist()[1],\
     camera2[cube_edges[0][1]].tolist()[2])

  for i in range(len(cube_edges)):
  # for i in [0]:
     print(camera2[cube_edges[i][0]].tolist()[0],camera2[cube_edges[i][0]].tolist()[1],\
     camera2[cube_edges[i][0]].tolist()[2],camera2[cube_edges[i][1]].tolist()[0],\
     camera2[cube_edges[i][1]].tolist()[1],camera2[cube_edges[i][1]].tolist()[2])
     ax.quiver(camera1[cube_edges[i][0]].tolist()[0],camera1[cube_edges[i][0]].tolist()[1],\
     camera1[cube_edges[i][0]].tolist()[2],camera1[cube_edges[i][1]].tolist()[0],\
     camera1[cube_edges[i][1]].tolist()[1],camera1[cube_edges[i][1]].tolist()[2],color=color[i])
     ax.quiver(camera2[cube_edges[i][0]].tolist()[0],camera2[cube_edges[i][0]].tolist()[1],\
     camera2[cube_edges[i][0]].tolist()[2],camera2[cube_edges[i][1]].tolist()[0]-camera2[cube_edges[i][0]].tolist()[0],\
     camera2[cube_edges[i][1]].tolist()[1]-camera2[cube_edges[i][0]].tolist()[1],camera2[cube_edges[i][1]].tolist()[2]-camera2[cube_edges[i][0]].tolist()[2],color=color[i])
  plt.show()

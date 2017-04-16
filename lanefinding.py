import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pprint

DEBUG = 1

def debug(msg):
  if DEBUG > 0:
    print(msg)

# TODO:
# [ ] save imgpoints array to reuse calibration on repeated runs

pp = pprint.PrettyPrinter(indent = 2)

img_bgr = mpimg.imread("camera_cal/calibration4.jpg")
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
imgshape = img_gray.shape[::-1]

# coordinate points in real world (no distortion)
worldpoints = []
# coordinate points on image (distortion)
imgpoints = []

# initialize vector for object points to zero
# this represents the real-world checkboard (all points to zero, no distortion)
obj = np.zeros((9*6,3), np.float32)
# and reshape vector into a 9x6 matrix
obj[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# read in all our prepared calibration images:
calibrationImages = glob.glob("camera_cal/calibration*.jpg")

cnt = 0;

for filename in calibrationImages:
  debug("reading " + filename)
  img_bgr = mpimg.imread(filename)

  # convert image from RGB to grayscale to help in finding corners
  img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
  # use cv2 function to detect chessboard corners (how helpful! we don't have to write that!)
  ret, corners = cv2.findChessboardCorners(img_gray, (9,6), None)

  if ret == True:
    cnt += 1
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    # TODO: refine corner subpix
    imgpoints.append(corners)
    # the worldpoints always get the perfect, real-world checkboard points:
    worldpoints.append(obj)
    #img_bgr = cv2.drawChessboardCorners(img_bgr, (9,6), corners,ret)
    #cv2.imshow('img',img_bgr)
    #cv2.waitKey(500)


debug("found corners in " + str(cnt) + " of " + str(len(calibrationImages)) + " calibration images")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(worldpoints, imgpoints, imgshape, None, None)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, imgshape, 1, imgshape)

# Test img
img = mpimg.imread("camera_cal/calibration5.jpg")
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

mpimg.imsave("undist.jpg", dst)
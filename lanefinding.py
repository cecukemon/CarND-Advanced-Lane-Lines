import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pprint
from moviepy.editor import VideoFileClip

### GLOBALS ###

# print debug output
DEBUG = 0

# number of corners in X direction
NX = 9
# number of corners in Y direction
NY = 6

# pretty printer object for debug purposes
pp = pprint.PrettyPrinter(indent = 2)

### FUNCTIONS ###

# print debug output if DEBUG
def debug(msg):
  if DEBUG > 0:
    print(msg)

# TODO:
# [ ] save imgpoints array to reuse calibration on repeated runs


# Calibrate camera, returning all parameters we need for cv2.undistort
def calibrate_camera():

  # get image shape from random calib image (assuming they're all same size)
  img_bgr = mpimg.imread("camera_cal/calibration4.jpg")
  img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
  imgshape = img_gray.shape[::-1]

  # coordinate points in real world (no distortion)
  worldpoints = []
  # coordinate points on image (distortion)
  imgpoints = []

  # initialize vector for object points to zero
  # this represents the real-world checkboard (all points to zero, no distortion)
  obj = np.zeros((NX*NY,3), np.float32)
  # and reshape vector into a 9x6 matrix
  obj[:,:2] = np.mgrid[0:NX,0:NY].T.reshape(-1,2)

  # read in all our prepared calibration images:
  calibrationImages = glob.glob("camera_cal/calibration*.jpg")

  cnt = 0;

  for filename in calibrationImages:
    debug("reading " + filename)
    img_bgr = mpimg.imread(filename)

    # convert image from RGB to grayscale to help in finding corners
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # use cv2 function to detect chessboard corners (how helpful! we don't have to write that!)
    ret, corners = cv2.findChessboardCorners(img_gray, (NX,NY), None)

    if ret == True:
      cnt += 1
      # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
      # TODO: refine corner subpix
      imgpoints.append(corners)
      # the worldpoints always get the perfect, real-world checkboard points:
      worldpoints.append(obj)

  debug("found corners in " + str(cnt) + " of " + str(len(calibrationImages)) + " calibration images")
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(worldpoints, imgpoints, imgshape, None, None)
  newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, imgshape, 1, imgshape)
  debug("camera calibration done")

  return(mtx, dist, newcameramtx)

# Save calibrated image to help debugging:
#def save_calib_img():
  # Test img
  #img = mpimg.imread("camera_cal/calibration5.jpg")
  #dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

  # Crop result
  #x,y,w,h = roi
  #dst = dst[y:y+h, x:x+w]
  #cv2.imwrite('undist.jpg',dst)

def unwarp_picture(img):

  # transform undistorted pic to grayscale:
  img_gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
  # save shape info for later:
  imgshape = img_gray.shape[::-1]

  # source:
  src = np.float32([ [315,685], [565,455], [700,455], [1130,685] ])
  # destination: fit unwarped picture nicely to img format
  offset = 90
  dst = np.float32([[offset, offset], [imgshape[0] - offset, offset], 
                                     [imgshape[0] - offset, imgshape[1] - offset], 
                                     [offset, imgshape[1] - offset]])

  M = cv2.getPerspectiveTransform(src, dst)
  warped = cv2.warpPerspective(img_undist, M, imgshape)

  return(warped, M)

def detect_lane_pixels(img):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary



def process_image(image):

    # use camera calibration parameters to undistort picture:
  img_undist = cv2.undistort(img, mtx, dist, newcameramtx)

  image2 = detect_lane_pixels
  (image3, M) = unwarp_picture(image2)

  return image3

### MAIN ###

# Calibrate camera:
mtx, dist, newcameramtx = calibrate_camera()


#vidcap = cv2.VideoCapture('project_video.mp4')
#success,image = vidcap.read()
#cv2.imwrite("frame.jpg" , image) 

#success = True
#while success:
#  success,image = vidcap.read()

output_filename = 'output.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(process_image)
output_clip.write_videofile(output_filename, audio=False)

# ...
# iterate over video images
#   [x] unwarp each frame
#   [ ] apply sobel and color correction to create threshholded binary image
#   [ ] perspective transform
#   [ ] identify lane pixels and fit with polynomial






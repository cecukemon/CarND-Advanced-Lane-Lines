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

# checkboard for calibration:
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

  # save shape info for later:
  imgshape = img.shape[::-1]

  # source - coordinates taken from still frame with straight road part:
  src = np.float32([ [560,460], [730,460], [1100,680], [230,680]])
  dst = np.float32([ [230,460], [1100,460], [1100,680], [230,680]])

  # calculate transformation matrix
  M = cv2.getPerspectiveTransform(src, dst)
  # calculate the inverse matrix as well, going to need this later
  Minv = cv2.getPerspectiveTransform(dst, src)
  # warp the image
  warped = cv2.warpPerspective(img, M, imgshape)

  return (warped, M, Minv)

# threshholding - directional gradient
def abs_sobel_thresh(direction, sobelx, sobely, thresh):

    if direction == 'x':
        abs_sobel = np.copy(sobelx)
    if direction == 'y':
        abs_sobel = np.copy(sobely)

    rescaled = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # initialize all-black image
    img_out = np.zeros_like(rescaled)

    # set all pixels that are between the threshhold min and max values to 255 (white)
    img_out[(rescaled >= thresh[0]) & (rescaled <= thresh[1])] = 255

    return img_out

# threshholding - gradient magnitude
def mag_threshhold(sobelx, sobely, thresh):

    grad = np.sqrt(sobelx**2 + sobely**2)

    scale = np.max(grad)/255 
    grad = (grad / scale).astype(np.uint8) 

    # see comments in abs_sobel_thresh
    img_out = np.zeros_like(grad)
    img_out[(grad >= thresh[0]) & (grad <= thresh[1])] = 255

    return img_out

# threshholding - gradient direction
def dir_threshold(sobelx, sobely, thresh):

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # see comments in abs_sobel_thresh
    img_out =  np.zeros_like(absgraddir)
    img_out[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255

    return img_out

# threshholding - color gradient, S channel (saturation)
def color_threshhold_s(img, thresh):

  # convert to HLS color space
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  s_channel = hls[:,:,2]

  # initialize empty (black) image
  img_out = np.zeros_like(s_channel)
  # and set all pixels that fall between threshhold boundaries
  img_out[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 255

  return img_out

def threshholding(img):
    img2 = np.copy(img)

    # turn image into greyscale
    gray_x = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    gray_y = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    #cv2.imwrite('frame_gray.jpg',gray)

    # sobel kernel size, larger is smoother
    ksize = 3

    # preprocess sobel kernel on grayscale image for both directions
    sx = cv2.Sobel(gray_x, cv2.CV_64F, 1, 0, ksize)
    sy = cv2.Sobel(gray_y, cv2.CV_64F, 0, 1, ksize)

    # calculate images with various threshhold filters
    # threshhold gradient x / gradient y
    grad_x = abs_sobel_thresh('x', sx, sy, thresh=(20, 100))
    grad_y = abs_sobel_thresh('y', sx, sy, thresh=(20, 100))
    # threshhold magniture
    mag_t = mag_threshhold(sx, sy, thresh=(20, 100))
    # threshhold direction
    dir_t = dir_threshold(sx, sy, thresh=(0.7, 1.3))

    #cv2.imwrite('frame_grad_x.png',grad_x)
    #cv2.imwrite('frame_grad_y.png',grad_y)
    #cv2.imwrite('frame_mag_t.png', mag_t)
    #cv2.imwrite('frame_dir_t.png', dir_t)

    # and combine the filters for a better result:
    combined = np.zeros_like(dir_t)
    combined[((grad_x == 255) & (grad_y == 255)) | ((mag_t == 255) & (dir_t == 255))] = 255

    color_t = color_threshhold_s(img2, thresh=(170,255))

    cv2.imwrite('frame_color_t.png', color_t)

    # combine with color threshhold result:
    combined2 = np.zeros_like(dir_t)
    combined2[(combined == 255) | (color_t == 255)] = 255

    return combined2

def find_lines(binary_warped):

  # calculate histogram on lower half of image to find the peaks (lots of white pixels)
  index = np.int(binary_warped.shape[0]/2)
  histogram = np.sum(binary_warped[index:,:], axis=0)

  # prepare empty output image
  out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

  # find the left and right peaks
  midpoint = np.int(histogram.shape[0]/2)
  leftx_base = np.argmax(histogram[:midpoint])
  rightx_base = np.argmax(histogram[midpoint:]) + midpoint

  # setup sliding windows
  nwindows = 9
  window_height = np.int(binary_warped.shape[0]/nwindows)
  nonzero = binary_warped.nonzero()
  nonzeroy = np.array(nonzero[0])
  nonzerox = np.array(nonzero[1])
  leftx_current = leftx_base
  rightx_current = rightx_base
  margin = 100
  minpix = 50
  left_lane_inds = []
  right_lane_inds = []

  # iterate over windows array:
  for window in range(nwindows):
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

  left_lane_inds = np.concatenate(left_lane_inds)
  right_lane_inds = np.concatenate(right_lane_inds)

  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds] 
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds] 

  # meter per pixels, correction factor
  ym_per_pix = 30/720
  xm_per_pix = 3.7/700

  # fit second order polynomial to left lane pixels
  left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
  # and calculate x coordinate
  left_fitx = left_fit_cr[0]*lefty**2 + left_fit_cr[1]*lefty + left_fit_cr[2]
  # same for right lane pixels
  right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
  right_fitx = right_fit_cr[0]*righty**2 + right_fit_cr[1]*righty + right_fit_cr[2]

  return (lefty, righty, left_fitx, right_fitx, left_fit_cr, right_fit_cr)

def measure_curvature(lefty, righty, left_fitx, right_fitx, left_fit_cr, right_fit_cr):

  # flip x side upside down to match y side
  #leftx = leftx[::-1]
  #rightx = rightx[::-1]

  y_eval = np.max(lefty)
  
  # use the polynomial we calculated earlier (in the left_fit_cr / right_fit_cr variables)
  # to calculate the curve radius according to:
  # http://www.intmath.com/applications-differentiation/8-radius-curvature.php
  left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
  right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

  return()

def draw_lane_on_image(original_image, warped_image, lefty, left_fitx, righty, right_fitx, Minv):

  # prepare empty (black) warped image:
  warp_zero = np.zeros_like(warped_image).astype(np.uint8)
  color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

  # transform x and y lane boundaries so they can be used with fillPoly
  pts_left = np.array([np.transpose(np.vstack([left_fitx, lefty]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, righty])))])
  pts = np.hstack((pts_left, pts_right))

  # draw lane poly:
  cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

  # use the inverse transformation matrix to unwarp the image
  newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0])) 
  # combine lane poly with the original image:
  result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

  return result

def process_image(image):

  # use camera calibration parameters to undistort picture:
  img_undist = cv2.undistort(image, mtx, dist, newcameramtx)

  #cv2.imwrite('frame_calib.png', img_undist)

  # Threshholding
  image2 = threshholding(img_undist)

  #cv2.imwrite('frame_threshholded.png',image2)

  # Perspective transform lane lines to straighten them
  (image3, M, Minv) = unwarp_picture(image2)

  (lefty, righty, left_fitx, right_fitx, left_fit_cr, right_fit_cr) = find_lines(image3)
  measure_curvature(lefty, righty, left_fitx, right_fitx, left_fit_cr, right_fit_cr)

  #img_test = cv2.cvtColor(img_undist, cv2.COLOR_RGB2GRAY)
  image4 = draw_lane_on_image(img_undist, image3, lefty, left_fitx, righty, right_fitx, Minv)

  return image4

### MAIN ###

# Calibrate camera:
mtx, dist, newcameramtx = calibrate_camera()

# read and process video:
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(process_image)

# write processed video with lane lines:
output_clip.write_videofile('output.mp4', audio=False)






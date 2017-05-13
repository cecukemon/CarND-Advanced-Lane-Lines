

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./img_for_writeup/undist.jpg "Undistorted calibration image"
[image2]: ./img_for_writeup/frame_calib.png "Undistorted video frame"
[image3]: ./img_for_writeup/frame_grad_x.png "Threshholding gradient in X"
[image4]: ./img_for_writeup/frame_color_t.png "Threshholding color"
[image5]: ./img_for_writeup/frame_threshholded.png "Threshholding combined"
[image6]: ./img_for_writeup/frame_transform.png "Transformed frame"
[image7]: ./img_for_writeup/frame_with_lanes.png "Frame with lanes drawn on"


[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The calibration code is in the calibrate_camera() function. First, I computed the image shape from a random calibration image, assuming all calibration images have the same dimensions. The image shape will be needed later. Then, I initialized a vector for the object points, which are the coordinates of the chessboard corners in the real-world - perfect with no camera distortion. The z coordinate is always 0, as the chessboard is flat.

Then I iterated over the directory with the calibration images, transforming each image to grayscale and using cv2.findChessboardCorners to detect chessboard corners. Some images did not successfully detect corners, I skipped those. If corners were detected, I appended those to the array imgpoints, and appended the precomputed, ideal corner coordinates to the array worldpoints.

Finally, I used cv2.calibrateCamera to calibrate the camera, and cv2.getOptimalNewCameraMatrix to compute a camera matrix that I will later use to undistort images. This is an example of an distortion-corrected calibration image:

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.

The distortion correction on the video frames happens in the process_image function. I call cv2.undistort on the frame, using the parameters computed during the camera calibration phase. Here's an example of the distortion-corrected image:

![alt text][image2]


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

For the threshholding, I wrote a function threshholding that makes a grayscale copy of the input image, and precomputes the Sobel kernel in x and y, then calls various other threshholding functions to compute different versions of the threshholding. The resulting images are then combined for the final result. I tried gradient threshholding in x and y direction (function abs_sobel_thresh), threshholding by gradient magnitude (mag_threshhold), threshholding by gradient direction (dir_threshhold) and finally, threshholding by color gradient on the S (saturation) channel (color_threshhold)

All those functions have the same basic structure: they compute the threshhold gradient, initialize an empty (black) image in the right size, then set those pixels that fall between the given threshhold boundaries, to one.

Here's an example of threshholding gradient in X:
![alt text][image3]

Here's an example of threshholding color:
![alt text][image4]

Here's an example of the combined images, using all computed threshholds:
![alt text][image5]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To do the perspective transform, I wrote a function unwarp_picture which uses cv2.getPerspectiveTransform and cv2.warpPerspective to warp the image. For the src and dst points, I took a still frame from the video from a section where the road was straight, and measured the pixel positions, and hardcoded them.

  src = np.float32([ [560,460], [730,460], [1100,680], [230,680]])
  dst = np.float32([ [230,460], [1100,460], [1100,680], [230,680]])

Here's an example of the transformed frame:
![alt text][image6]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To identify the lane pixels, I used a histogram to detect the start, and sliding windows to follow the line. I computed a histogram of the pixels on the lower half of the transformed image, adding up the pixels to find the starting point for the sliding windows. The peaks are likely the left and the right lane starting point. I then set up an array of sliding windows to follow the peaks on each iteration. Finally, I fitted the line points with a polynomial using the numpy polyfit function.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the curvature radius in the function measure_curvature. 

The calculated 

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function draw_lane_on_image


![alt text][image7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_2.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Steps where I had issues:
- wasted a lot of time debugging issues due to wrong file format because I had saved some test frames as PNG instead of JPG, and the alpha channel made for some weird bugs
- had some problems figuring out the perspective transform steps


Things that could be improved:
- pickle and save camera calibration data between program runs to save time (I'm working on an antique Macbook air)
- understand why the camera calibration on real images leads to weird color
- try more treshholding parameters and combinations of filters and optimize the rsult
- the perspective transform results are suboptimal
- didn't have time to implement a class to keep track of relevant parameters between frames


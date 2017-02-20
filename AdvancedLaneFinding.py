import os
import pickle
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import find_peaks_cwt
import time
### Calibration


#test
fname = 'camera_cal/calibration{}.jpg'.format(2)
img = cv2.imread(fname)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
nx = 9
ny = 6

#Test to find chessboard corners

def test_find_chessboard_corners(img,show_img):
    objpoints_tmp = []
    imgpoints_tmp = []

    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        imgpoints_tmp.append(corners)
        objpoints_tmp.append(objp)
        if show_img:
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #plt.imshow(img)
            #plt.show()
            plt.savefig("tmp/test_find_chessboard_corners.jpg")
    else:
        print("ret == False")

    return objpoints_tmp, imgpoints_tmp

objpoints, imgpoints = test_find_chessboard_corners(img,True)

#camera calibration, undistort
def test_calibration_undistort(img, show_img):

    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist

    if show_img:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        ax1.imshow(img)
        ax1.set_title("Origin Image", fontsize=30)
        ax2.imshow(dst)
        ax2.set_title("Undistorted Image", fontsize=30)
        plt.show()

    return dst,mtx,dist

dst,mtx,dist = test_calibration_undistort(img,False)
img = dst


# warp
def warp(img, draw_img=False):
    img_size = (img.shape[1], img.shape[0])
    print(img_size)
    src = np.float32([[152,172],[1206,184],[262,638],[1064,626]])
    #dst = np.float32([[128,72],[1280-128,72],[128,720-72],[1280-128,720-72]])
    edge_y = 116
    edge_x = 160
    dst = np.float32([[edge_x, edge_y], [1280 - edge_x, edge_y], [edge_x, 720 - edge_y], [1280 - edge_x, 720 - edge_y]])
    M = cv2.getPerspectiveTransform(src,dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    if draw_img:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Source Image')
        ax1.imshow(img)
        ax2.set_title('Warped image')
        ax2.imshow(warped)
        plt.show()

    return warped

warped_im = warp(img, False)


#color and gradient
def color_and_gradient():

    test_image = mpimg.imread("test_images/test1.jpg")
    hls = cv2.cvtColor(test_image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx / np.max(abs_sobelx))
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    s_thresh_min =170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary==1) | (sxbinary == 1)] = 1

    f, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    ax1.set_title("stacked threshold")
    ax1.imshow(color_binary)

    ax2.set_title('Combined S channel and gradient threshold')
    ax2.imshow(combined_binary, cmap='gray')

    plt.show()

#color_and_gradient()

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if(len(img.shape)>2):
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


#Sobel threshold
def threshold_abs_sobel(gray, orient='x', sobel_kernel = 3, thresh = (0, 255)):
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    #Absolute sobel value
    sobel = np.absolute(sobel)

    #Scaled Sobel value
    scaled_sobel = np.uint8(255*sobel / np.max((sobel)))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return binary_output

#Mag threshold
def threshold_mag(gray, sobel_kernel = 3, mag_thresh = (0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255 * sobel / np.max(sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    return binary_output

#Define a function to threshold an image for a given range and Sobel kernel
def threshold_direction(gray, sobel_kernel = 3, thresh=(0, np.pi / 2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize =  sobel_kernel)

    try:
        absgraddir = np.absolute(np.arctan(sobely / sobelx))
        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir > thresh[0]) & (absgraddir < thresh[1])] = 1
    except ZeroDivisionError:
        print("zero division error in threshold direction")
    except:
        print("Other errors in threshold direction")

    return dir_binary


def pipeline(img):
    #Gaussian Blur
    kernel_size = 5
    img = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)

    #HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    #Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #sobel kernel
    ksize = 7
    gradx = threshold_abs_sobel(gray, orient ='x', sobel_kernel=ksize, thresh=(10, 255))
    grady = threshold_abs_sobel(gray, orient ='y', sobel_kernel=ksize, thresh =(60, 255))
    mag_binary = threshold_mag(gray, sobel_kernel=ksize, mag_thresh=(40, 255))
    dir_binary = threshold_direction(gray, sobel_kernel=ksize, thresh=(.65, 1.05))
    combined = np.zeros_like(dir_binary)
    combined[ ((gradx == 1) & (grady == 1)) |  ((mag_binary == 1) & (dir_binary == 1)) ] = 1
    s_binary = np.zeros_like(combined)
    s_binary[(s>160) & (s<255)] = 1
    color_binary = np.zeros_like(combined)
    color_binary[(s_binary > 0) | (combined > 0)] = 1
    imshape = img.shape
    left_bottom = (100, imshape[0])
    right_bottom = (imshape[1]-20, imshape[0])
    apex1 = (610, 410)
    apex2 = (680, 410)
    inner_left_bottom = (310, imshape[0])
    inner_right_bottom = (1150, imshape[1])
    inner_apex1 = (700, 480)
    inner_apex2 = (650, 480)
    vertices = np.array([[left_bottom, apex1, apex2, right_bottom, inner_right_bottom, inner_apex1, inner_apex2, inner_left_bottom]], dtype=np.int32)

    color_binary = region_of_interest(color_binary, vertices)

    return color_binary


show_img = False
#Process the test images
for i in range(1,7):
    fname_short = "test{}.jpg".format(i)
    fname = "test_images/" + fname_short
    image = cv2.imread(fname)
    result = pipeline(image)

    if show_img:

        f, (ax1, ax2) = plt.subplots(1,2, figsize=(24,9))
        f.tight_layout()

        ax1.imshow(image)
        ax1.set_title('Origin Image', fontsize=40)

        ax2.imshow(result, cmap='gray')
        ax2.set_title('Result', fontsize=40)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        plt.show()



#Warping

image_shape = image.shape
print("Image Shape = ", image_shape)


area_of_interest = [[150 + 430, 460], [1150 - 440, 460], [1150, 720], [150, 720]]

def corners_unwarp(img, nx, ny , mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    offset1 = 200
    offset2 = 0
    offset3 = 0

    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #img_size = (gray.shape[1], gray.shape[0])
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(area_of_interest)
    dst = np.float32([[offset1, offset3], [img_size[0] - offset1, offset3], [img_size[0] - offset1, img_size[1] - offset2], [offset1, img_size[1]-offset2]])

    M = cv2.getPerspectiveTransform(src, dst)
    MinV = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, img_size)

    return warped, M, MinV

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        self.windows = np.ones((3,12))*-1

def calculate_curvature(yvals, fitx):
    y_eval = np.max(yvals)
    ym_per_pix = 30/ 720
    xm_per_pix = 3.7/700
    fit_cr = np.polyfit(yvals*ym_per_pix, fitx*xm_per_pix, 2)
    curverad = ((1+(2*fit_cr[0]*y_eval + fit_cr[1]) ** 2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad

def find_position(pts):
    position = image_shape[1] / 2
    left = np.min( pts[ (pts[:,1] < position) & (pts[:,0] > 700)][:,1])
    right = np.max(pts[ (pts[:,1] > position) & (pts[:,0] > 700)][:,1])
    center = (left + right) / 2
    xm_per_pix = 3.7 / 700
    return (position - center) * xm_per_pix

def find_nearest(array,value):
    # Function to find the nearest point from array
    if len(array) > 0:
        idx = (np.abs(array-value)).argmin()
        return array[idx]

def find_peaks(image, y_window_top, y_window_bottom, x_left, x_right):
    # Find the historgram from the image inside the window
    histogram = np.sum(image[y_window_top:y_window_bottom, :], axis=0)
    # Find the max from the histogram
    if len(histogram[int(x_left):int(x_right)]) > 0:
        return np.argmax(histogram[int(x_left):int(x_right)]) + x_left
    else:
        return (x_left + x_right) / 2

def sanity_check(lane, curverad, fitx, fit):
    # Sanity check for the lane
    if lane.detected:  # If lane is detected
        # If sanity check passes
        if abs(curverad / lane.radius_of_curvature - 1) < .6:
            lane.detected = True
            lane.current_fit = fit
            lane.allx = fitx
            lane.bestx = np.mean(fitx)
            lane.radius_of_curvature = curverad
            lane.current_fit = fit
        # If sanity check fails use the previous values
        else:
            lane.detected = False
            fitx = lane.allx
    else:
        # If lane was not detected and no curvature is defined
        if lane.radius_of_curvature:
            if abs(curverad / lane.radius_of_curvature - 1) < 1:
                lane.detected = True
                lane.current_fit = fit
                lane.allx = fitx
                lane.bestx = np.mean(fitx)
                lane.radius_of_curvature = curverad
                lane.current_fit = fit
            else:
                lane.detected = False
                fitx = lane.allx
                # If curvature was defined
        else:
            lane.detected = True
            lane.current_fit = fit
            lane.allx = fitx
            lane.bestx = np.mean(fitx)
            lane.radius_of_curvature = curverad
    return fitx

# Sanity check for the direction
def sanity_check_direction(right, right_pre, right_pre2):
    # If the direction is ok then pass
    if abs((right-right_pre) / (right_pre-right_pre2) - 1) < .2:
        return right
    # If not then compute the value from the previous values
    else:
        return right_pre + (right_pre - right_pre2)

# find_lanes function will detect left and right lanes from the warped image.
# 'n' windows will be used to identify peaks of histograms
def find_lanes(n, image, x_window, lanes, \
               left_lane_x, left_lane_y, right_lane_x, right_lane_y, window_ind):
    # 'n' windows will be used to identify peaks of histograms
    # Set index1. This is used for placeholder.
    index1 = np.zeros((n + 1, 2))
    index1[0] = [300, 1100]
    index1[1] = [300, 1100]
    # Set the first left and right values
    left, right = (300, 1100)
    # Set the center
    center = 700
    # Set the previous center
    center_pre = center
    # Set the direction
    direction = 0
    for i in range(n - 1):
        # set the window range.
        y_window_top = 720 - 720 / n * (i + 1)
        y_window_bottom = 720 - 720 / n * i
        # If left and right lanes are detected from the previous image
        if (left_lane.detected == False) and (right_lane.detected == False):
            # Find the historgram from the image inside the window
            left = find_peaks(image, y_window_top, y_window_bottom, index1[i + 1, 0] - 200, index1[i + 1, 0] + 200)
            right = find_peaks(image, y_window_top, y_window_bottom, index1[i + 1, 1] - 200, index1[i + 1, 1] + 200)
            # Set the direction
            left = sanity_check_direction(left, index1[i + 1, 0], index1[i, 0])
            right = sanity_check_direction(right, index1[i + 1, 1], index1[i, 1])
            # Set the center
            center_pre = center
            center = (left + right) / 2
            direction = center - center_pre
        # If both lanes were detected in the previous image
        # Set them equal to the previous one
        else:
            left = left_lane.windows[window_ind, i]
            right = right_lane.windows[window_ind, i]
        # Make sure the distance between left and right laens are wide enough
        if abs(left - right) > 600:
            # Append coordinates to the left lane arrays
            left_lane_array = lanes[(lanes[:, 1] >= left - x_window) & (lanes[:, 1] < left + x_window) &
                                    (lanes[:, 0] <= y_window_bottom) & (lanes[:, 0] >= y_window_top)]
            left_lane_x += left_lane_array[:, 1].flatten().tolist()
            left_lane_y += left_lane_array[:, 0].flatten().tolist()
            if not math.isnan(np.mean(left_lane_array[:, 1])):
                left_lane.windows[window_ind, i] = np.mean(left_lane_array[:, 1])
                index1[i + 2, 0] = np.mean(left_lane_array[:, 1])
            else:
                index1[i + 2, 0] = index1[i + 1, 0] + direction
                left_lane.windows[window_ind, i] = index1[i + 2, 0]
            # Append coordinates to the right lane arrays
            right_lane_array = lanes[(lanes[:, 1] >= right - x_window) & (lanes[:, 1] < right + x_window) &
                                     (lanes[:, 0] < y_window_bottom) & (lanes[:, 0] >= y_window_top)]
            right_lane_x += right_lane_array[:, 1].flatten().tolist()
            right_lane_y += right_lane_array[:, 0].flatten().tolist()
            if not math.isnan(np.mean(right_lane_array[:, 1])):
                right_lane.windows[window_ind, i] = np.mean(right_lane_array[:, 1])
                index1[i + 2, 1] = np.mean(right_lane_array[:, 1])
            else:
                index1[i + 2, 1] = index1[i + 1, 1] + direction
                right_lane.windows[window_ind, i] = index1[i + 2, 1]
    return left_lane_x, left_lane_y, right_lane_x, right_lane_y


import math


# Function to find the fitting lines from the warped image
def fit_lanes(image):
    # define y coordinate values for plotting
    yvals = np.linspace(0, 100, num=101) * 7.2  # to cover same y-range as image
    # find the coordinates from the image
    lanes = np.argwhere(image)
    # Coordinates for left lane
    left_lane_x = []
    left_lane_y = []
    # Coordinates for right lane
    right_lane_x = []
    right_lane_y = []
    # Curving left or right - -1: left 1: right
    curve = 0
    # Set left and right as None
    left = None
    right = None
    # Find lanes from three repeated procedures with different window values
    left_lane_x, left_lane_y, right_lane_x, right_lane_y \
        = find_lanes(4, image, 25, lanes, \
                     left_lane_x, left_lane_y, right_lane_x, right_lane_y, 0)
    left_lane_x, left_lane_y, right_lane_x, right_lane_y \
        = find_lanes(6, image, 50, lanes, \
                     left_lane_x, left_lane_y, right_lane_x, right_lane_y, 1)
    left_lane_x, left_lane_y, right_lane_x, right_lane_y \
        = find_lanes(8, image, 75, lanes, \
                     left_lane_x, left_lane_y, right_lane_x, right_lane_y, 2)
    # Find the coefficients of polynomials
    left_fit = np.polyfit(left_lane_y, left_lane_x, 2)
    left_fitx = left_fit[0] * yvals ** 2 + left_fit[1] * yvals + left_fit[2]
    right_fit = np.polyfit(right_lane_y, right_lane_x, 2)
    right_fitx = right_fit[0] * yvals ** 2 + right_fit[1] * yvals + right_fit[2]
    # Find curvatures
    left_curverad = calculate_curvature(yvals, left_fitx)
    right_curverad = calculate_curvature(yvals, right_fitx)
    # Sanity check for the lanes
    left_fitx = sanity_check(left_lane, left_curverad, left_fitx, left_fit)
    right_fitx = sanity_check(right_lane, right_curverad, right_fitx, right_fit)

    return yvals, left_fitx, right_fitx, left_lane_x, left_lane_y, right_lane_x, right_lane_y, left_curverad


# draw poly on an image
# def draw_poly(image, warped, yvals, left_fitx, right_fitx, Minv):
def draw_poly(image, warped, yvals, left_fitx, right_fitx,
              left_lane_x, left_lane_y, right_lane_x, right_lane_y, Minv, curvature):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    # Put text on an image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Radius of Curvature: {} m".format(int(curvature))
    cv2.putText(result,text,(400,100), font, 1,(255,255,255),2)
    # Find the position of the car
    pts = np.argwhere(newwarp[:,:,1])
    position = find_position(pts)
    if position < 0:
        text = "Vehicle is {:.2f} m left of center".format(-position)
    else:
        text = "Vehicle is {:.2f} m right of center".format(position)
    cv2.putText(result,text,(400,150), font, 1,(255,255,255),2)
    return result

# This function will color the image
# Input: Original image
# Output: Original image with colored region
def process_image(image):
    # Apply pipeline to the image to create black and white image
    img = pipeline(image)
    # Warp the image to make lanes parallel to each other
    top_down, perspective_M, perspective_Minv = corners_unwarp(img, nx, ny, mtx, dist)
    # Find the lines fitting to left and right lanes
    a, b, c, lx, ly, rx, ry, curvature = fit_lanes(top_down)
    # Return the original image with colored region
    return draw_poly(image, top_down, a, b, c, lx, ly, rx, ry, perspective_Minv, curvature)




#Process Image

x_values = [area_of_interest[0][0], area_of_interest[1][0], area_of_interest[2][0], area_of_interest[3][0], area_of_interest[0][0]]
y_values = [area_of_interest[0][1], area_of_interest[1][1], area_of_interest[2][1], area_of_interest[3][1], area_of_interest[0][1]]

for i in range(1,7):

    left_lane = Line()
    right_lane = Line()

    fname_short = "test{}.jpg".format(i)
    fname = "test_images/" + fname_short

    img_raw = cv2.imread(fname)
    img = pipeline(img_raw)

    top_down, perspective_M, perspective_Minv = corners_unwarp(img, nx, ny, mtx, dist)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(24,9))
    f.tight_layout()

    ax1.set_title('Gray Image', fontsize = 20)
    ax1.plot(x_values, y_values, 'r-', lw=2)
    ax1.imshow(img, cmap = 'gray')

    a,b,c,_,_,_,_,_ = fit_lanes(top_down)
    ax2.plot(b,a, color='green', linewidth=5)
    ax2.plot(c,b, color='blue', linewidth=5)
    ax2.imshow(top_down, cmap='gray')
    ax2.set_title("Undistorted and warped", fontsize=20)

    left_lane = Line()
    right_lane = Line()

    image_color = process_image(img_raw)
    ax3.imshow(image_color)
    ax3.set_title('Image with a color', fontsize=20)
    top_down[top_down > 0] = 1
    histogram = np.sum(top_down[:240, :], axis = 0)
    ax4.plot(histogram)
    histogram = np.sum(top_down[240:480,:], axis = 0)
    ax4.plot(histogram)
    histogram=np.sum(top_down[480:,:], axis=0)
    ax4.plot(histogram)

    indexes = find_peaks_cwt(histogram, np.arange(1,550))
    ax4.set_title('histogram')

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    # Save processed test image into output images folder
    plt.savefig("output_images/processed_" + fname_short)

    #plt.show()
    #time.sleep(3)
    plt.close()


##Video

### Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
# Set up lines for left and right
left_lane = Line()
right_lane = Line()
white_output = 'project_video_processed.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)







import os
import pickle
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

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

def crop_region_of_interest(img):
    outer_left_bottom =   (100, 720)
    outer_apex1 =         (610, 410)
    outer_apex2 =         (680, 410)
    outer_right_bottom =  (1260,720)
    inner_right_bottom =  (1150,720)     #1150,1280 #1150,720
    inner_apex1 =         (700, 480)
    inner_apex2 =         (650, 480)
    inner_left_bottom =   (300, 720)


    vertices = np.array([[outer_left_bottom, outer_apex1, outer_apex2, outer_right_bottom, inner_right_bottom, inner_apex1, inner_apex2, inner_left_bottom]], dtype=np.int32)

    mask = np.zeros_like(img)
    ignore_mask_color = (255,) * 3
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

def pipeline(image):

    #GaussianBlur with kernel size = 5
    image = cv2.GaussianBlur(image,(5, 5),0)

    #HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]

    #Gray color
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #Sobel threshold
    sobelx = threshold_abs_sobel(gray, orient ='x', sobel_kernel=7, thresh=(10, 255))
    sobely = threshold_abs_sobel(gray, orient ='y', sobel_kernel=7, thresh =(60, 255))

    #Mag threshold
    ksize = 7
    mag_binary = threshold_mag(gray, sobel_kernel=ksize, mag_thresh=(40, 255))

    #Direction threshold
    dir_binary = threshold_direction(gray, sobel_kernel=ksize, thresh=(.65, 1.05))

    #Combine the 3 thresholds
    combined = np.zeros_like(dir_binary)
    combined[ ((sobelx == 1) & (sobely == 1)) |  ((mag_binary == 1) & (dir_binary == 1)) ] = 1

    s_binary = np.zeros_like(combined)
    s_binary[(s>160) & (s<255)] = 1

    color_binary = np.zeros_like(combined)
    color_binary[(s_binary > 0) | (combined > 0)] = 1

    color_binary = crop_region_of_interest(color_binary)

    return color_binary

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


def corners_unwarp(img, nx, ny , mtx, dist):
    area_of_interest = [[580, 460], [710, 460], [1150, 720], [150, 720]]

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    offset1 = 200
    offset2 = 0
    offset3 = 0

    img_size = (img.shape[1], img.shape[0])
    src = np.float32(area_of_interest)
    #                [200, 0],             [1080,0],                         [1080, 720] ,                                     [200, 720]
    dst = np.float32([[offset1, offset3], [img_size[0] - offset1, offset3], [img_size[0] - offset1, img_size[1] - offset2], [offset1, img_size[1]-offset2]])
    M = cv2.getPerspectiveTransform(src, dst)
    MinV = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, img_size)

    return warped, M, MinV

nx = 9
ny = 6

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
            plt.imshow(img)
            #plt.show()
            plt.savefig("tmp/test_find_chessboard_corners.jpg")
    else:
        print("ret == False")

    return objpoints_tmp, imgpoints_tmp

fname = 'camera_cal/calibration{}.jpg'.format(2)
img = cv2.imread(fname)
objpoints, imgpoints = test_find_chessboard_corners(img,True)

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

# Sanity check for the direction, make sure the direction won't turbulent too much
def stable_direction(right, right_pre1, right_pre2):
    # If the direction not change too much
    if abs((right-right_pre1) / (right_pre1-right_pre2) - 1) < .2:
        return right
    # If not, then compute the value from the previous values
    else:
        return right_pre1 + (right_pre1 - right_pre2)

def find_lanes(binary_warped, img, perspective_Minv):
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    #plt.plot(histogram * 1e305)

    nwindows = 9
    window_height = np.int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    leftx_current_pre1 = leftx_current
    leftx_current_pre2 = leftx_current
    rightx_current_pre1 = rightx_current
    rightx_current_pre2 = rightx_current



    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            leftx_current = stable_direction(leftx_current, leftx_current_pre1, leftx_current_pre2)
            leftx_current_pre1 = leftx_current
            leftx_current_pre2 = leftx_current_pre1
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            rightx_current = stable_direction(rightx_current, rightx_current_pre1, rightx_current_pre2)
            rightx_current_pre1 = rightx_current
            rightx_current_pre2 = rightx_current_pre1

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    y_eval = np.max(ploty)
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) * 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])



    print(left_curverad, 'm', right_curverad, 'm')


    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzeroy[right_lane_inds]] = [0, 0, 255]


    # ------------skip sliding window
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy
                                   + left_fit[2] - margin)) & (
                      nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy
                                    + right_fit[2] - margin)) & (
                       nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # -----

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    '''
    left_line_window1 = np.array(          [np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.flipud(np.transpose(np.vstack([right_fitx - margin, ploty])))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

    '''

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    right_line_pts = np.hstack((left_line_window1, right_line_window2))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    '''
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()
    '''

    #New
    #warp_zero = np.zeros_like(result).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    newwarp = cv2.warpPerspective(result, perspective_Minv, (result.shape[1], result.shape[0]))

    img = img.astype(np.uint8)
    newwarp = newwarp.astype(np.uint8)
    result2 = cv2.addWeighted(img, 1, newwarp, 0.8, 0)

    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - result.shape[1]/2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'


    #draw curveness
    text = "Curvature: {} m".format(int(left_curverad))
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(result2, text, (100, 100), font, 1, (255, 255, 255), 2)

    text_offcenter = "Vehicle is " + str(abs(round(center_diff,3))) + ' m ' + side_pos + ' of center '
    cv2.putText(result2, text_offcenter, (100, 150), font, 1, (255, 255, 255), 2)
    #plt.imshow(img)
    #plt.show()

    #plt.imshow(newwarp)
    #plt.show()

    #plt.imshow(result2)
    #plt.show()

    return result2,histogram

def fit_lanes(top_down, img, perspective_Minv):
    return  find_lanes(top_down, img, perspective_Minv)



''''''

def process_image(rawimg):
    img = pipeline(rawimg)

    top_down, perspective_M, perspective_Minv = corners_unwarp(img, nx, ny, mtx, dist)

    #plt.imshow(img)
    #plt.show()

    #plt.imshow(top_down)
    #plt.show()

    #plt.imshow(top_down)
    #plt.show()

    return fit_lanes(top_down, rawimg, perspective_Minv)
    #a, b, c, lx, ly, rx, ry, curvature = fit_lanes(top_down, rawimg,perspective_Minv)
    #image_color = draw_poly(image, top_down, a, b, c, lx, ly, rx, ry, perspective_Minv, curvature)

def process_image_for_image(rawimg):
    return process_image(rawimg)

def process_image_for_video(rawimg):
    return process_image(rawimg)[0]

for i in range(1,7):

    left_lane = Line()
    right_lane = Line()

    fname_short = "test"+ str(i) +".jpg"
    fname = "test_images/" + fname_short

    rawimg = cv2.imread(fname)

    image_color, histogram = process_image_for_image(rawimg)

    #draw image
    f, (ax1, ax2) = plt.subplots(1,2,figsize=(24,12))
    f.tight_layout()

    ax1.set_title('Result', fontsize = 20)
    ax1.imshow(image_color)
    ax1.set_title('Histogram', fontsize = 20)
    ax2.plot(histogram * 1e305)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig("output_images/new_processed_" + fname_short)
    plt.close()

    #plt.imshow(image_color)
    #plt.show()

from moviepy.editor import VideoFileClip
#import pygame





white_output = 'project_video_processed.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image_for_video)
print(type(white_clip))
white_clip.preview()
white_clip.write_videofile(white_output, audio=False)

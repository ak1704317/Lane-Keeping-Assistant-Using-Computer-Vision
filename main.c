# advanced_lane_full.py
# Advanced ADAS-style lane detection (white lane on black road)
# - Bird's-eye (perspective) transform (interactive tuning)
# - Binary (gradient + intensity) for white lines
# - Sliding windows + polynomial fit
# - Lane center & steering error
#
# Requirements: python3-picamera2, python3-opencv, numpy
# Run: python3 advanced_lane_full.py

from picamera2 import Picamera2
import cv2
import numpy as np
import time
from collections import deque

# --------------------- PARAMETERS / SMOOTHING ---------------------
FRAME_W, FRAME_H = 640, 480
SMOOTHING_FRAMES = 5   # smoothing length for polynomial coefficients
YM_PER_PIX = 30/480.0  # (example) meters per pixel in y — tune if you want meters
XM_PER_PIX = 3.7/640.0 # (example) meters per pixel in x — tune for real scale (not critical for small robot)

# --------------------- Camera init ---------------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (FRAME_W, FRAME_H)}))
picam2.start()
time.sleep(0.2)

# --------------------- UI / Trackbars ---------------------
cv2.namedWindow("Warp")
cv2.namedWindow("Binary")
cv2.namedWindow("Output")

def nothing(x): pass

# Perspective trapezoid parameters (percentages)
cv2.createTrackbar("TopY%", "Warp", 35, 80, nothing)         # top edge height (percent of image)
cv2.createTrackbar("TopWidth%", "Warp", 50, 100, nothing)    # top width as percent of frame width
cv2.createTrackbar("BotInset%", "Warp", 10, 40, nothing)     # bottom inset from sides (percent)
# Binary tuning
cv2.createTrackbar("ThreshWhite", "Binary", 200, 255, nothing)
cv2.createTrackbar("SobelK", "Binary", 3, 31, nothing)       # must be odd
cv2.createTrackbar("SobelTh", "Binary", 50, 255, nothing)

# --------------------- Helper functions ---------------------
def get_perspective_mats(w, h):
    """Create src/dst perspective matrices using trackbar values."""
    topy_pct = cv2.getTrackbarPos("TopY%", "Warp")
    topw_pct = cv2.getTrackbarPos("TopWidth%", "Warp")
    bot_inset_pct = cv2.getTrackbarPos("BotInset%", "Warp")

    top_y = int(h * topy_pct / 100.0)
    top_w = int(w * topw_pct / 100.0)
    bot_inset = int(w * bot_inset_pct / 100.0)

    src = np.float32([
        [(w - top_w)//2, top_y],
        [(w + top_w)//2, top_y],
        [bot_inset, h - 1],
        [w - 1 - bot_inset, h - 1]
    ])
    dst = np.float32([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv, src

def binary_from_frame(frame):
    """Produce a binary image where white lane pixels = 255."""
    # 1) grayscale + blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # 2) intensity threshold for bright (white) lane
    t_white = cv2.getTrackbarPos("ThreshWhite", "Binary")
    _, white_bin = cv2.threshold(blur, t_white, 255, cv2.THRESH_BINARY)

    # 3) Sobel X to emphasize vertical edges (useful for thin lines)
    sob_k = cv2.getTrackbarPos("SobelK", "Binary")
    if sob_k % 2 == 0: sob_k = max(1, sob_k-1)
    sob = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=sob_k)
    sob_abs = np.absolute(sob)
    sob_norm = np.uint8(255 * sob_abs / (np.max(sob_abs) + 1e-6))
    sob_th = cv2.getTrackbarPos("SobelTh", "Binary")
    _, sob_bin = cv2.threshold(sob_norm, sob_th, 255, cv2.THRESH_BINARY)

    # 4) Combine intensity and gradient
    combined = cv2.bitwise_or(white_bin, sob_bin)

    # 5) Morphology to clean
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    return combined

def sliding_windows_poly(binary_warped, nwindows=9, margin=50, minpix=50):
    """Detect left & right lane using sliding windows. Returns fits and pixel positions."""
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]//2)

    # Try to detect peaks for left and right
    leftx_base = np.argmax(histogram[:midpoint]) if np.any(histogram[:midpoint]) else None
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint if np.any(histogram[midpoint:]) else None

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if leftx_base is None and rightx_base is None:
        return None, None, None, None

    # sliding windows
    window_height = int(binary_warped.shape[0]//nwindows)
    left_current = leftx_base
    right_current = rightx_base

    left_inds = []
    right_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        if left_current is not None:
            win_xleft_low = left_current - margin
            win_xleft_high = left_current + margin
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            left_inds.append(good_left_inds)
            if len(good_left_inds) > minpix:
                left_current = int(np.mean(nonzerox[good_left_inds]))

        if right_current is not None:
            win_xright_low = right_current - margin
            win_xright_high = right_current + margin
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            right_inds.append(good_right_inds)
            if len(good_right_inds) > minpix:
                right_current = int(np.mean(nonzerox[good_right_inds]))

    # flatten
    left_inds = np.concatenate(left_inds) if left_inds and len(left_inds)>0 else np.array([], dtype=int)
    right_inds = np.concatenate(right_inds) if right_inds and len(right_inds)>0 else np.array([], dtype=int)

    # if no pixels found, return None
    if left_inds.size == 0:
        left_fit = None
    else:
        leftx = nonzerox[left_inds]; lefty = nonzeroy[left_inds]
        left_fit = np.polyfit(lefty, leftx, 2)

    if right_inds.size == 0:
        right_fit = None
    else:
        rightx = nonzerox[right_inds]; righty = nonzeroy[right_inds]
        right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, left_inds, right_inds

# smoothing buffers for fits
left_fit_buffer = deque(maxlen=SMOOTHING_FRAMES)
right_fit_buffer = deque(maxlen=SMOOTHING_FRAMES)

def smooth_fit(buffer, fit):
    if fit is None:
        return None
    buffer.append(fit)
    arr = np.array(buffer)
    return np.mean(arr, axis=0)

# --------------------- Main loop ---------------------
try:
    while True:
        frame = picam2.capture_array()
        M, Minv, src_pts = get_perspective_mats(FRAME_W, FRAME_H)

        # 1) binary image
        binary = binary_from_frame(frame)

        # 2) warp to bird's-eye
        warped = cv2.warpPerspective(binary, M, (FRAME_W, FRAME_H), flags=cv2.INTER_LINEAR)

        # 3) sliding window & poly fit
        left_fit, right_fit, left_inds, right_inds = sliding_windows_poly(warped, nwindows=9, margin=50, minpix=50)

        # 4) smooth fits
        smooth_left = smooth_fit(left_fit_buffer, left_fit) if left_fit is not None else (left_fit_buffer.clear() or None)
        smooth_right = smooth_fit(right_fit_buffer, right_fit) if right_fit is not None else (right_fit_buffer.clear() or None)

        # 5) compute lane center
        y_eval = warped.shape[0] - 1
        lane_center_x = None
        if smooth_left is not None and smooth_right is not None:
            left_x = smooth_left[0]*y_eval**2 + smooth_left[1]*y_eval + smooth_left[2]
            right_x = smooth_right[0]*y_eval**2 + smooth_right[1]*y_eval + smooth_right[2]
            lane_center_x = (left_x + right_x) / 2.0
            method = "both"
        elif smooth_left is not None:
            left_x = smooth_left[0]*y_eval**2 + smooth_left[1]*y_eval + smooth_left[2]
            lane_center_x = left_x + 200  # heuristic offset if only left found (tune)
            method = "left_only"
        elif smooth_right is not None:
            right_x = smooth_right[0]*y_eval**2 + smooth_right[1]*y_eval + smooth_right[2]
            lane_center_x = right_x - 200  # heuristic offset if only right found (tune)
            method = "right_only"
        else:
            lane_center_x = None
            method = "none"

        # 6) compute error relative to image center (in pixels)
        img_center = FRAME_W / 2.0
        if lane_center_x is not None:
            error_px = img_center - lane_center_x
            # convert to meters if you want (requires xm_per_pix calibration)
            error_m = error_px * XM_PER_PIX
        else:
            error_px = None
            error_m = None

        # ---------------- Visualization ----------------
        # Warp the original frame for overlay
        warped_color = cv2.warpPerspective(frame, M, (FRAME_W, FRAME_H))

        out_img = warped_color.copy()

        # draw detected lane area
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0], dtype=np.int32)
        if smooth_left is not None:
            left_fitx = (smooth_left[0]*ploty**2 + smooth_left[1]*ploty + smooth_left[2]).astype(np.int32)
            for i in range(len(ploty)-1):
                cv2.line(out_img, (left_fitx[i], ploty[i]), (left_fitx[i+1], ploty[i+1]), (0,255,0), 2)
        if smooth_right is not None:
            right_fitx = (smooth_right[0]*ploty**2 + smooth_right[1]*ploty + smooth_right[2]).astype(np.int32)
            for i in range(len(ploty)-1):
                cv2.line(out_img, (right_fitx[i], ploty[i]), (right_fitx[i+1], ploty[i+1]), (0,255,0), 2)

        # draw center line and lane center marker
        cv2.line(out_img, (int(img_center),0), (int(img_center), FRAME_H), (255,0,0), 2)
        if lane_center_x is not None:
            cv2.circle(out_img, (int(lane_center_x), FRAME_H-30), 8, (0,0,255), -1)
            cv2.putText(out_img, f"Err(px): {int(error_px)}  Err(m): {error_m:.3f}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
            cv2.putText(out_img, f"Method: {method}", (10,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        else:
            cv2.putText(out_img, "Lane Lost", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # show source trapezoid on original for tuning
        src_pts_int = src_pts.astype(np.int32)
        vis_src = frame.copy()
        cv2.polylines(vis_src, [src_pts_int], True, (0,255,255), 2)

        cv2.imshow("Binary", warped)
        cv2.imshow("Warp", vis_src)
        cv2.imshow("Output", out_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(f"warp_{ts}.png", warped)
            cv2.imwrite(f"out_{ts}.png", out_img)

finally:
    cv2.destroyAllWindows()
    picam2.close()    SO IS THIS CODE  FULL final Advanced ADAS polynomial lane detection code?
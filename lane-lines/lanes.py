import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_image(image):
    # Create the grayscale
    gray_scaled = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply Gausian Blur
    blurred = cv2.GaussianBlur(gray_scaled, (5, 5), 0)
    # Apply amd return the Canny function
    return cv2.Canny(blurred, 50, 150)

def region_of_interest(image):
    height, _ = image.shape
    # Define the region of interest
    triad = np.array([[(200, height), (1100, height), (550, 250)]])
    # Create a matrix of zeroes
    mask = np.zeros_like(image)
    # Fill the pixels in the region with white pixels
    cv2.fillPoly(mask, triad, 255)
    # Perform a bitwise and on the image and the mask, so only the pixels in the
    # region of interest remain untouched, while pixels out of the region of interest
    # are black (0)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def make_coords(image, params):
    slope, intercept = params
    height, _, _ = image.shape

    y1, y2 = height, int(height * (3 / 5))
    x1, x2 = int((y1 - intercept) / slope), int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left, right = [], []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)

        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))
    
    left_average, right_average = np.average(left, axis = 0), np.average(right, axis = 0)
    left_line, right_line = make_coords(image, left_average), make_coords(image, right_average)

    return np.array([left_line, right_line])

def find_lane_lanes(image):
    return cv2.HoughLinesP(image, 2, np.pi / 180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)

def add_found_lines(original, traced):
    zeroed_image = np.zeros_like(original)

    if traced is not None:
        for x1, y1, x2, y2 in traced:
            cv2.line(zeroed_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return zeroed_image

def detect_lane_lines(image):
    # Copy the image
    lane_image = np.copy(image)

    cannied = canny_image(lane_image)
    cropped = region_of_interest(cannied)
    lines = find_lane_lanes(cropped)
    drawn_lines = add_found_lines(lane_image, average_slope_intercept(lane_image, lines))

    return cv2.addWeighted(lane_image, 0.8, drawn_lines, 1, 1)

captured = cv2.VideoCapture("test2.mp4")

while captured.isOpened():
    _, frame = captured.read()
    detected_result = detect_lane_lines(frame)

    cv2.imshow('result', detected_result)

    if cv2.waitKey(1) == ord('q'):
        break
    
captured.release()
cv2.destroyAllWindows()



import cv2
import numpy as np

def imageCopy(src):
    return np.copy(src)

def makeBlackImage(image, color=False):
    height, width = image.shape[0], image.shape[1]
    if color is True:
        return np.zeros((height, width, 3), np.uint8)
    else:
        if len(image.shape) == 2:
            return np.zeros((height, width), np.uint8)
        else:
            return np.zeros((height, width, 3), np.uint8)

def convertColor(image, flag=cv2.COLOR_BGR2GRAY):
    return cv2.cvtColor(image, flag)
    
def rangeColor(image, lower, upper):
    result = imageCopy(image)
    return cv2.inRange(result, lower, upper)
    
def addImage(image1, image2):
    return cv2.add(image1, image2)
    
def imageMorphologyKernel(flag=cv2.MORPH_RECT, size=5):
    return cv2.getStructuringElement(flag, (size, size))
    
def imageMorphologyEx(image, op, kernel, iterations=1):
    return cv2.morphologyEx(image, op=op, kernel=kernel, iterations=iterations)

def fillPolyROI(image, points):
    if len(image.shape) == 2:
        channels = 1
    else:
        channels = image.shape[2]
    mask = makeBlackImage(image)
    ignore_mask_color = (255,) * channels
    cv2.fillPoly(mask, points, ignore_mask_color)
    return mask

def polyROI(image, points):
    mask = fillPolyROI(image, points)
    return cv2.bitwise_and(image, mask)

def houghLinesP(image, rho=1.0, theta=np.pi/180, threshold=100, minLineLength=10, maxLineGap=100):
    return cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

def splitTwoSideLines(lines, slope_threshold = (5. * np.pi / 180.)):
    lefts = []
    rights = []
    for line in lines:
        x1 = line[0,0]
        y1 = line[0,1]
        x2 = line[0,2]
        y2 = line[0,3]
        if (x2-x1) == 0:
            continue
        slope = (float)(y2-y1)/(float)(x2-x1)
        if abs(slope) < slope_threshold:
            continue
        if slope <= 0:
            lefts.append([slope, x1, y1, x2, y2])
        else:
            rights.append([slope, x1, y1, x2, y2])
    return lefts, rights

def medianPoint(x):
    if len(x) == 0:
        return None
    else:
        xx = sorted(x)
        return xx[(int)(len(xx)/2)]

def interpolate(x1, y1, x2, y2, y):
    return int(float(y - y1) * float(x2-x1) / float(y2-y1) + x1)

def lineFitting(image, lines, color = (0,0,255), thickness = 3, slope_threshold = (5. * np.pi / 180.)):
    result = imageCopy(image)
    height = image.shape[0]
    lefts, rights = splitTwoSideLines(lines, slope_threshold)
    left = medianPoint(lefts)
    right = medianPoint(rights)
    min_y = int(height*0.6)
    max_y = height
    min_x_left = interpolate(left[1], left[2], left[3], left[4], min_y)
    max_x_left = interpolate(left[1], left[2], left[3], left[4], max_y)
    min_x_right = interpolate(right[1], right[2], right[3], right[4], min_y)
    max_x_right = interpolate(right[1], right[2], right[3], right[4], max_y)
    cv2.line(result, (min_x_left, min_y), (max_x_left, max_y), color, thickness)
    cv2.line(result, (min_x_right, min_y), (max_x_right, max_y), color, thickness)
    return result

def lane_detection_and_draw(image):
    result = imageCopy(image)
    HLS = convertColor(result, cv2.COLOR_BGR2HLS)
    Y_lower = np.array([15, 52, 75])
    Y_upper = np.array([30, 190, 255])
    Y_BIN = rangeColor(HLS, Y_lower, Y_upper)
    W_lower = np.array([0, 200, 0])
    W_upper = np.array([180, 255, 255])
    W_BIN = rangeColor(HLS, W_lower, W_upper)
    result = addImage(Y_BIN, W_BIN)
    MORPH_ELLIPSE = imageMorphologyKernel(cv2.MORPH_ELLIPSE, 7)
    result = imageMorphologyEx(result, cv2.MORPH_CLOSE , MORPH_ELLIPSE)    
    MORPH_CROSS = imageMorphologyKernel(cv2.MORPH_CROSS, 3)
    result = imageMorphologyEx(result, cv2.MORPH_OPEN , MORPH_CROSS)
    result_line = imageMorphologyEx(result, cv2.MORPH_GRADIENT , MORPH_CROSS)
    height, width = image.shape[:2]
    src_pt1 = [int(width*0.4), int(height*0.65)]
    src_pt2 = [int(width*0.6), int(height*0.65)]
    src_pt3 = [int(width*0.9), int(height*0.9)]
    src_pt4 = [int(width*0.1), int(height*0.9)]
    roi_poly_02 = np.array([[tuple(src_pt1), tuple(src_pt2), tuple(src_pt3), tuple(src_pt4)]], dtype=np.int32)
    line_roi = polyROI(result_line, roi_poly_02)
    lines = houghLinesP(line_roi, 1, np.pi/180, 10, 5, 10)
    result = lineFitting(image, lines, (0, 0, 255), 5, 5. * np.pi / 180.)
    return result

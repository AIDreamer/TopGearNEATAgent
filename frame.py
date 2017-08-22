import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
from functools import reduce
import math


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def get_monitor_frame(img):
    # resize the image using specific ratio
    # this will achieve better result than using image blurring
    ratio = 0.02
    resized = cv2.resize(img, (0,0), fx=ratio, fy=ratio)
    #img = cv2.GaussianBlur(img, (3, 3), 0)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 3)

    edged = auto_canny(blurred)

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    _, cnts, _= cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    print(cv2.contourArea(cnts[0]))

    screenCnt = None
    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is not None:
        # reconstruct the original image
        p1, p2, p3, p4 = screenCnt
        pts1 = np.float32((p1, p4, p2, p3))
        # 320 x 180
        pts2 = np.float32([[0,0],[320,0],[0,180],[320,180]])

        M = cv2.getPerspectiveTransform(pts1,pts2)

        dst = cv2.warpPerspective(resized,M,(320, 180))
    plt.subplot(121),plt.imshow(resized),plt.title('Input')
    plt.subplot(122),plt.imshow(edged),plt.title('Output')
    plt.show()
    return get_main_screen(dst)

def color_distance(color_map, color2):
    result = [[0 for i in range(len(color_map[0]))] for j in range(len(color_map))]
    for i in range(len(color_map)):
        for j in range(len(color_map[0])):
            color = color_map[i][j]
            diff = math.sqrt((color[0] - color2[0]) ** 2 + (color[1] - color[1]) ** 2 + (color[2] - color2[2]) ** 2)
            result[i][j] = diff
    return np.array(result)

def get_main_screen(screen):
    WHITE = (255, 255, 255)
    points = []
    # search top points
    for i in range(len(screen)):
        if i in range(len(screen) // 5, len(screen) // 5 * 4) or i in range(20) or i in range(len(screen) - 20, len(screen)):
            continue # skip the middle part
        for j in range(len(screen[i])):
            if j in range(len(screen[i]) // 5, len(screen[i]) // 5 * 4):
                continue # skip the middle part
            if color_distance(screen[i][j], WHITE) > 200:
                points.append((i, j))
    box = cv2.boundingRect(np.array(points))
    print(box)


def brute_force_screen(img):
    WHITE = (255, 255, 255)
    ratio = 0.05
    resized = cv2.resize(img, (0,0), fx=ratio, fy=ratio)

    mid_x, mid_y = len(resized) // 2, len(resized[0]) // 2
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)

    dimg = color_distance(blurred, WHITE)
    #print(dimg)
    point1 = None
    for i in range(mid_y, 1, -1):
        p1, p2 = dimg[mid_x][i], dimg[mid_x][i-1]
        if p1 < 100 and p2 < 100:
            point1 = (mid_x, i)
            break

    point2 = None
    for i in range(mid_y, len(resized[0]) - 1):
        p1, p2 = dimg[mid_x][i], dimg[mid_x][i+1]
        if p1 < 100 and p2 < 100:
            point2 = (mid_x, i)
            break

    point3 = None
    for i in range(mid_x, -1, -1):
        p0, p1, p2 = dimg[i][mid_y-1], dimg[i][mid_y], dimg[i][mid_y-1]
        if sum([p0, p1, p2]) / 3 < 200:
            point3 = (i, mid_y)
            break

    point4 = None
    for i in range(mid_x, len(resized)):
        p0, p1, p2 = dimg[i][mid_y-1], dimg[i][mid_y], dimg[i][mid_y-1]
        if sum([p0, p1, p2]) / 3 < 200:
            point4 = (i, mid_y)
            break

    row_start = int(point3[0] / ratio)
    row_end = int(point4[0] / ratio)
    col_start = int(point1[1] / ratio)
    col_end = int(point2[1] / ratio)
    result = img[row_start:row_end, col_start:col_end]

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(result),plt.title('Output')
    plt.show()
    return result

def get_frame(filename):
    vid = imageio.get_reader(filename,  'ffmpeg')
    image = vid.get_data(50)

    #get_monitor_frame(image)
    brute_force_screen(image)

if __name__ == "__main__":
    #img = cv2.imread("img1.jpg")
    #get_monitor_frame(img)

    get_frame("capture.mp4")

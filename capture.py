import cv2
import time
import numpy as np
from extract import mark, print_map
import requests
import json
import math

points = []

# ---------
# CONSTANTS
# ---------

CAR_BOX = (104, 71, 48, 28)
IMG_SHAPE = (224, 256)

TOPO_SHAPE = (28, 32)
INTERVAL = 4
interval_count = 0
main_topo = None

sand_stall = 0
x, y, w, h = CAR_BOX
RATIO_ROW = IMG_SHAPE[0] // TOPO_SHAPE[0]
RATIO_COL = IMG_SHAPE[1] // TOPO_SHAPE[1]
CAR_ROW_START = math.ceil(y / RATIO_ROW)
CAR_COL_START = math.ceil(x / RATIO_COL)
SHAPE_COL = math.floor(w / RATIO_COL)
SHAPE_ROW = math.floor(h / RATIO_ROW)


def draw_points(event, x, y, flags, param):
    global points
    if len(points) >= 4:
        return
    if event == cv2.EVENT_LBUTTONUP:
        points.append((x, y))

# inspired from http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
	    [0, 0],
	    [maxWidth - 1, 0],
	    [maxWidth - 1, maxHeight - 1],
	    [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def read_frame():
    global points
    # produce two pictures
    cv2.namedWindow("input")
    cv2.namedWindow("output")
    cv2.setMouseCallback("input", draw_points)

    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        original = frame.copy()
        for x, y in points:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("input", frame)

        # compute the output
        if len(points) == 4:
            # proceed to transform
            pts = order_points(np.array(points))
            transformed = four_point_transform(original, pts)
            cv2.imshow("output", transformed)
            yield transformed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    for frame in read_frame():
        interval_count += 1

        if interval_count == INTERVAL:
            main_topo = mark(frame)[:TOPO_SHAPE[0]//2, :]
            main_topo[CAR_ROW_START : CAR_ROW_START + SHAPE_ROW, CAR_COL_START : CAR_COL_START + SHAPE_COL] = False
            main_topo = main_topo.flatten().astype(int)
            topo_json = json.dumps({"Topo": np.array_str(main_topo)})
            interval_count = 0
            r = requests.post("http://127.0.0.1:37979/cam_topo", data = topo_json)
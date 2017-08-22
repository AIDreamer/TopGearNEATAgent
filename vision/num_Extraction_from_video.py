import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt
import imageio
from functools import reduce
import math
import time

root = "../img_frame/"
dirs = ["nitro/", "rank/", "time/","speed/"]

source_image = []

NITRO_FRAME = (24,80,10,9)
SPEED_FRAME_1 = (183,16,15,12)
SPEED_FRAME_2 = (198,16,15,12)
SPEED_FRAME_3 = (214,16,15,12)
TIME_FRAME_1 = (184,33,8,7)
TIME_FRAME_2 = (192,33,8,7)
TIME_FRAME_3 = (208,33,8,7)
TIME_FRAME_4 = (216,33,8,7)
TIME_FRAME_5 = (232,33,8,7)
TIME_FRAME_6 = (240,33,8,7)
RANK_FRAME = (201,88,32,13)


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


    # plt.subplot(121),plt.imshow(img),plt.title('Input')
    # plt.subplot(122),plt.imshow(result),plt.title('Output')
    # plt.show()


    return result


def get_frame(filename):
    vid = imageio.get_reader(filename,  'ffmpeg')
    image = vid.get_data(70)
    #get_monitor_frame(image)
    return brute_force_screen(image)


def setup_img(source):
    for dir in dirs:
        # Find all the image name
        file_list = [f for f in listdir(root + dir) if isfile(join(root + dir, f))]

        newlist = []
        # For name in file list
        for file in file_list:
            img = cv2.imread(root + dir + file, cv2.IMREAD_GRAYSCALE)
            newlist = newlist + [img]

        source = source + [newlist]

# convertion for rank and nitro
def convertImage(rawimage):
    for i in range(len(rawimage)):
        for j in range(len(rawimage[0])):
            distColor = [0,0,0]
            distColor[0] = rawimage[i][j] - 0
            distColor[1] = abs(rawimage[i][j] - 128)
            distColor[2] = 255 - rawimage[i][j]
            mindist = min(distColor)

            if (distColor.index(mindist) == 0):
                rawimage[i][j] = 0

            if (distColor.index(mindist) == 1):
                rawimage[i][j] = 128

            if (distColor.index(mindist) == 2):
                rawimage[i][j] = 255

# convertion for speed images
def convertImageSpeed(rawimage):
    for i in range(len(rawimage)):
        for j in range(len(rawimage[0])):
            distColor = [0,0,0]
            distColor[0] = abs(rawimage[i][j] - 0)
            distColor[1] = abs(rawimage[i][j] - 47)
            distColor[2] = abs(rawimage[i][j] - 76)
            mindist = min(distColor)

            if (distColor.index(mindist) == 0):
                rawimage[i][j] = 0

            if (distColor.index(mindist) == 1):
                if rawimage[i][j] >= 52:
                    rawimage[i][j] = 76
                else:
                    rawimage[i][j] = 47

            if (distColor.index(mindist) == 2):
                rawimage[i][j] = 76





if __name__ == "__main__":
    #img = cv2.imread("img1.jpg")
    #get_monitor_frame(img)

    img = get_frame("capture.mp4")
    img = img[28:, 28:]
    img = cv2.resize(img, (256, 112))

    # nitro
    nx, ny, nw, nh = NITRO_FRAME
    nitro = img[ny:ny+nh, nx:nx+nw]
    nitro = cv2.cvtColor(nitro, cv2.COLOR_BGR2GRAY)
    convertImage(nitro)

    # plt.plot(), plt.imshow(nitro), plt.title('Output')
    # plt.show()
    # print(nitro)

    # rank
    rx, ry, rw, rh = RANK_FRAME
    rank = img[ry:ry + rh, rx:rx + rw]
    rank = cv2.cvtColor(rank, cv2.COLOR_BGR2GRAY)
    convertImage(rank)

    # plt.plot(), plt.imshow(rank), plt.title('Output')
    # plt.show()
    # print(rank)

    # speed 1
    s1x, s1y, s1w, s1h = SPEED_FRAME_1
    speed1 = img[s1y:s1y + s1h, s1x:s1x + s1w]
    speed1 = cv2.cvtColor(speed1, cv2.COLOR_BGR2GRAY)
    convertImageSpeed(speed1)

    # plt.plot(), plt.imshow(speed1), plt.title('Output')
    # plt.show()
    # print(speed1)

    # speed 2
    s2x, s2y, s2w, s2h = SPEED_FRAME_2
    speed2 = img[s2y:s2y + s2h, s2x:s2x + s2w]
    speed2 = cv2.cvtColor(speed2, cv2.COLOR_BGR2GRAY)
    convertImageSpeed(speed2)

    # plt.plot(), plt.imshow(speed2), plt.title('Output')
    # plt.show()
    # print(speed2)

    # speed 3
    s3x, s3y, s3w, s3h = SPEED_FRAME_3
    speed3 = img[s3y:s3y + s3h, s3x:s3x + s3w]
    speed3 = cv2.cvtColor(speed3, cv2.COLOR_BGR2GRAY)
    convertImageSpeed(speed3)

    # plt.plot(), plt.imshow(speed3), plt.title('Output')
    # plt.show()
    # print(speed3)

    # Set up the source images
    for dir in dirs:

        file_list = [f for f in listdir(root + dir) if isfile(join(root + dir, f))]

        newlist = []
        # For name in file list
        for file in file_list:
            img1 = cv2.imread(root + dir + file)
            if dir == "speed/":
                img1 = img1[1:13,:15]
            imgcal = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            newlist = newlist + [imgcal]

        source_image = source_image + [newlist]

    # nitro determine
    n_dist = []
    for targetImg in source_image[0]:
        # only count the white parts

        # dist = 0
        # count = 0
        # for i in range(0,nw):
        #     for j in range(0,nh):
        #
        #         if (nitro[j][i] == 255) :
        #             count = count + 1
        #             dist = dist + (255 - targetImg[j][i]) ** 2
        #
        # dist = np.sqrt(dist)/count*nw*nh
        # n_dist.append(dist)

        # count the whole image
        n_dist.append(np.sqrt(np.sum((nitro - targetImg) ** 2)))

    print(n_dist)
    nitro_num = n_dist.index(min(n_dist)) + 1
    print("Determined nitro: " + str(nitro_num))


    # rank determine
    r_dist = []
    for targetImg in source_image[1]:
        # only count the white parts

        # plt.plot(), plt.imshow(targetImg), plt.title('Output')
        # plt.show()
        # dist = 0
        # count = 0
        # for i in range(0, rw):
        #     for j in range(0, rh):
        #
        #         if (rank[j][i] == 255):
        #             count = count + 1
        #             dist = dist + (255 - targetImg[j][i]) ** 2
        #
        # dist = np.sqrt(dist) / count * rw * rh
        # r_dist.append(dist)

        # count the whole image
        r_dist.append(np.sqrt(np.sum((rank - targetImg) ** 2)))

    print(r_dist)

    r_order = [10,11,12,13,14,15,16,17,18,19,1,20,2,3,4,5,6,7,8,9]
    rank_num = r_order[r_dist.index(min(r_dist))]
    print("Determined rank: " + str(rank_num))

    # speed1 determine
    s1_dist = []
    for targetImg in source_image[3]:
        # only count the red parts

        # dist = 0
        # count = 0
        # for i in range(0, s1w):
        #     for j in range(0, s1h):
        #
        #         if (speed1[j][i] == 76) or (speed1[j][i] == 47):
        #             count = count + 1
        #             if (targetImg[j][i] == 17):
        #                 dist = dist + (speed1[j][i] - 255) ** 2
        #             else:
        #                 dist = dist + (speed1[j][i] - targetImg[j][i]) ** 2
        #
        #         if (targetImg[j][i] == 76) or (targetImg[j][i] == 47):
        #             dist = dist + (speed1[j][i] - targetImg[j][i]) ** 2
        #
        # dist = np.sqrt(dist) / count * s1w * s1h
        # s1_dist.append(dist)

        # count the whole image
        s1_dist.append(np.sqrt(np.sum((speed1 - targetImg) ** 2)))

    print(s1_dist)
    speed1_num = s1_dist.index(min(s1_dist))
    print("Determined speed1: " + str(speed1_num))

    # speed2 determine
    s2_dist = []
    for targetImg in source_image[3]:

        # only count the red parts

        # dist = 0
        # count = 0
        # for i in range(0, s2w):
        #     for j in range(0, s2h):
        #
        #         if (speed2[j][i] == 76) or (speed2[j][i] == 47):
        #             count = count + 1
        #             if (targetImg[j][i] == 17):
        #                 dist = dist + (speed2[j][i] - 255) ** 2
        #             else:
        #                 dist = dist + (speed2[j][i] - targetImg[j][i]) ** 2
        #
        #         if (targetImg[j][i] == 76) or (targetImg[j][i] == 47):
        #             dist = dist + (speed2[j][i] - targetImg[j][i]) ** 2
        #
        # dist = np.sqrt(dist) / count * s2w * s2h
        # s2_dist.append(dist)

        # count the whole image
        s2_dist.append(np.sqrt(np.sum((speed2 - targetImg) ** 2)))

    print(s2_dist)
    speed2_num = s2_dist.index(min(s2_dist))
    print("Determined speed2: " + str(speed2_num))

    # speed3 determine
    s3_dist = []
    for targetImg in source_image[3]:

        # only count the red parts

        # dist = 0
        # count = 0
        # for i in range(0, s3w):
        #     for j in range(0, s3h):
        #
        #         if (speed3[j][i] == 76) or (speed3[j][i] == 47):
        #             count = count + 1
        #             if (targetImg[j][i] == 17):
        #                 dist = dist + (speed3[j][i] - 255) ** 2
        #             else:
        #                 dist = dist + (speed3[j][i] - targetImg[j][i]) ** 2
        #
        #         if (targetImg[j][i] == 76) or (targetImg[j][i] == 47):
        #             dist = dist + (speed3[j][i] - targetImg[j][i]) ** 2
        #
        # dist = np.sqrt(dist) / count * s3w * s3h
        # s3_dist.append(dist)

        # count the whole image
        s3_dist.append(np.sqrt(np.sum((speed3 - targetImg) ** 2)))

    print(s3_dist)
    speed3_num = s3_dist.index(min(s3_dist))
    print("Determined speed3: " + str(speed3_num))

    speed_total = speed1_num*100+speed2_num*10+speed3_num
    print("Determined speed: " + str(speed_total))








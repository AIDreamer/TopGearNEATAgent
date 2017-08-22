import cv2
from matplotlib import pyplot as plt
import math
import numpy
import os

def mark(img):
    ratio = 0.5

    TOPO_SHAPE = (32,28)
    img = cv2.resize(img, TOPO_SHAPE)

    # compute the distance
    # distance to road
    road_color1 = (41, 66, 86)
    road_color2 = (66, 82, 82)
    road_color3 = (206, 206, 206)
    road_color4 = (156, 0 ,0)

    other_color1 = (222, 173, 140)
    other_color2 = (189, 140, 107)
    other_color3 = (1, 87, 1)
    other_color4 = (1, 107, 1)

    road_map1 = color_distance(img, road_color1)
    road_map2 = color_distance(img, road_color2)
    road_map3 = color_distance(img, road_color3)
    road_map4 = color_distance(img, road_color4)
    #other_map1 = color_distance(img_map, other_color1)
    #other_map2 = color_distance(img_map, other_color2)
    #other_map3 = color_distance(img_map, other_color3)
    #other_map4 = color_distance(img_map, other_color4)

    maps = [road_map1, road_map2, road_map3, road_map4]
    bool_maps = []
    for m in maps:
        bool_maps.append(mark_bool(m, 50))

    road = bool_maps[0]
    for map in bool_maps[1:]:
        road = merge_map(road, map)
    road = road.astype(numpy.uint8) * 255
    return road

    # marked = mark_map(road)
    # result = downsample_map(marked)
    # return result

def expert_direction(topo):
    # compute the center of mass
    topo = topo > 0
    sum_col = 0
    sum_mass = 0
    TOP = 10 if len(topo) > 10 else 2

    for j in range(len(topo[0])):
        col_sum = sum(topo[:TOP, j])
        sum_mass += j * col_sum
        sum_col += col_sum
    col = sum_mass / sum_col

    car_center_left = len(topo[0]) // 2 - 2
    car_center_right = car_center_left + 4

    if col > car_center_right:
        # goes right
        return "0010000100"
    elif col < car_center_left:
        # goes left
        return "0010001000"
    else:
        # goes straight
        return "0010000000"

def print_map(m):
    os.system('cls')
    for i in range(len(m)):
        for j in range(len(m[i])):
            print("{0:2}".format(m[i][j]), end="")
        print()

def downsample_map(m):
    result = []
    i = 0
    while i < len(m) - 1:
        j = 0
        row = []
        while j < len(m[i]) - 1:
            batch = [m[i][j], m[i][j + 1], m[i + 1][j], m[i + 1][j + 1]]
            dic = {"0": batch.count("0"), "-": batch.count("-"), "": batch.count("")}
            if dic[""] > dic["0"] and dic[""] > dic["0"]:
                row.append("")
            else:
                if dic["0"] >= dic["-"]:
                    row.append("0")
                else:
                    row.append("-")

            #if m[i][j] == "0" or m[i+1][j] == "0" or m[i][j+1] == "0" or m[i+1][j+1] == "0":
            #    row.append("0")
            #elif m[i][j] == "-" or m[i+1][j] == "-" or m[i][j+1] == "-" or m[i+1][j+1] == "-":
            #    row.append("-")
            #else:
            #    row.append("")
            j += 2
        result.append(row)
        i += 2
    result = filter_map(numpy.array(result))
    return result

def filter_map(m):
    # filter out the noise, which might caused by something else
    result = m.copy()
    for i in range(len(result)):
        for j in range(1, len(result[i]) - 1):
            if result[i][j-1] == '-' and result[i][j] == '0' and result[i][j+1] == '-':
                result[i][j] = '-'
    return result



def mark_map(m):
    result = []
    for i in range(len(m)):
        has_hit = False
        row = []
        for j in range(len(m[i])):
            if not m[i][j]:
                if has_hit:
                    row.append("0") # car
                else:
                    row.append("")
            else:
                row.append("-") # road
                has_hit = True
        result.append(row)

    # reverse the scanning
    for i in range(len(m)):
        for j in range(len(m[i]) -1, -1, -1):
            if not m[i][j] and result[i][j] == '0':
                result[i][j] = ''
            elif m[i][j]:
                break

    return numpy.array(result)



def merge_map(map1, map2):
    result = []
    for i in range(len(map1)):
        row = []
        for j in range(len(map1[0])):
            row.append(map1[i][j] or map2[i][j])
        result.append(row)
    return numpy.array(result)

def mark_bool(m, threshold):
    result = []
    for i in range(len(m)):
        row = []
        for j in range(len(m[0])):
            row.append(m[i][j] < threshold)
        result.append(row)
    return numpy.array(result)


def color_distance(img, color):
    result = []
    c2 = color
    for i in range(len(img)):
        row = []
        for j in range(len(img[i])):
            c = img[i][j]
            d = math.sqrt((c[0] - c2[2]) ** 2 + (c[1] - c2[1]) ** 2 + (c[2] - c2[0]) ** 2)
            row.append(d)
        result.append(row)
    return numpy.array(result)

def test():
    img = cv2.imread("car.png")
    m = mark(img)
    print_map(m)
    b,g,r = cv2.split(img)       # get b,g,r
    img = cv2.merge([r,g,b])     # switch it to rgb
    plt.imshow(img)
    plt.show()
    pass


if __name__ == "__main__":
    test()

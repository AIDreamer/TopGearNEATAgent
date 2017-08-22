import cv2
from os import listdir
from os.path import isfile, join
import numpy as np

root = "../img_frame/"
dirs = ["nitro/", "rank/", "time/"]

for dir in dirs:
    # Find all the image name
    file_list = [f for f in listdir(root + dir) if isfile(join(root + dir, f))]

    # For name in file list
    for file in file_list:
        img = cv2.imread(root + dir + file, cv2.IMREAD_GRAYSCALE)
        mask = np.invert((img == 0) | (img == 255))
        img[mask] = 128
        cv2.imwrite(root + dir + file[:-4] + "_gray.png", img)
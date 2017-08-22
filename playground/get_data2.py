# -----------------------------------
# STEP 0: SOME CONSTANT AND LIBRARIES
# -----------------------------------

# Libraries
import ctypes
import numpy as np
import time

# Constants



GAME_TITLE = "SNES - Top Gear (USA)"
DEBUGGER_TITLE = "Graphics Debugger"

# Frame limit
FRAME_LIMIT = 10000

NITRO_FRAME = (23,80,10,9)
SPEED_FRAME_1 = (184,16,16,14)
SPEED_FRAME_2 = (200,16,16,14)
SPEED_FRAME_3 = (216,16,16,14)
TIME_FRAME_1 = (184,33,8,7)
TIME_FRAME_2 = (192,33,8,7)
TIME_FRAME_3 = (208,33,8,7)
TIME_FRAME_4 = (216,33,8,7)
TIME_FRAME_5 = (232,33,8,7)
TIME_FRAME_6 = (240,33,8,7)
RANK_FRAME = (200,91,32,13)

# --------------------------------------------------------------------------------------
# STEP 1: CHECK IF APPROPRIATE WINDOWS HAVE BEEN OPENED AND MOVE TO APPROPRIATE POSITION
# --------------------------------------------------------------------------------------

def get_window_titles():
    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    GetWindowText = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible

    titles = []

    def foreach_window(hwnd, lParam):
        if IsWindowVisible(hwnd):
            length = GetWindowTextLength(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(hwnd, buff, length + 1)
            titles.append((hwnd, buff.value))
        return True

    EnumWindows(EnumWindowsProc(foreach_window), 0)

    return titles

# Get all top level windows
window_titles = get_window_titles()

# Check if the required programs are there
check_game = False

for hwnd, title in window_titles:
    if GAME_TITLE == title:
        check_game = True
        game_handler = hwnd

assert(check_game)

ctypes.windll.user32.MoveWindow(game_handler, 0, 0, 272, 327, True)
ctypes.windll.user32.BringWindowToTop(game_handler)

# -----------------
# Stream video data
# -----------------

def save_img(img, frame_data, name):
    x, y, width, height = frame_data
    data = img[y : y+height, x : x+width]
    data = Image.fromarray(data)
    data.save(name)

from PIL import Image
import numpy as np
import cv2

road_stream = None
obj_stream = None

from PIL import ImageGrab, Image
import numpy as np
import cv2

OBJ_STREAM_TOP_X = 851
OBJ_STREAM_TOP_Y= 60

ROAD_TOP_X = 8
ROAD_TOP_Y = 73

frame_count = 0

while(True):

    start = time.time()
    road = ImageGrab.grab(bbox=(ROAD_TOP_X, ROAD_TOP_Y, ROAD_TOP_X + 256, ROAD_TOP_Y + 112)) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    end = time.time()
    print("Execution time = ", str(end - start))

    road = np.array(road)

    # save_img(road, NITRO_FRAME, "nitro_" + str(frame_count) + ".png")
    # save_img(road, SPEED_FRAME_1, "speed_1_" + str(frame_count) + ".png")
    # save_img(road, SPEED_FRAME_2, "speed_2_" + str(frame_count) + ".png")
    # save_img(road, SPEED_FRAME_3, "speed_3_" + str(frame_count) + ".png")
    # save_img(road, TIME_FRAME_1, "time_1_" + str(frame_count) + ".png")
    # save_img(road, TIME_FRAME_2, "time_2_" + str(frame_count) + ".png")
    # save_img(road, TIME_FRAME_3, "time_3_" + str(frame_count) + ".png")
    # save_img(road, TIME_FRAME_4, "time_4_" + str(frame_count) + ".png")
    # save_img(road, TIME_FRAME_5, "time_5_" + str(frame_count) + ".png")
    # save_img(road, TIME_FRAME_6, "time_6_" + str(frame_count) + ".png")
    save_img(road, RANK_FRAME, "rank_" + str(frame_count) + ".png")

    frame_count += 1
    if frame_count > FRAME_LIMIT: break
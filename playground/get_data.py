# -----------------------------------
# STEP 0: SOME CONSTANT AND LIBRARIES
# -----------------------------------

# Libraries
import ctypes
import numpy as np
import time

# Constants

TAKE_OBJECT = False
TAKE_ROAD = True

GAME_TITLE = "SNES - Top Gear (USA)"
DEBUGGER_TITLE = "Graphics Debugger"

# Frame limit
FRAME_LIMIT = 100

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
check_debugger = False
check_game = False

for hwnd, title in window_titles:
    if TAKE_ROAD and GAME_TITLE == title:
        check_game = True
        game_handler = hwnd
    if TAKE_OBJECT and DEBUGGER_TITLE == title:
        check_debugger = True
        debugger_handler = hwnd

assert(TAKE_ROAD or check_debugger)
assert(TAKE_OBJECT or check_game)

if TAKE_OBJECT: ctypes.windll.user32.MoveWindow(debugger_handler, 275, 0, 1114, 746, True)
if TAKE_OBJECT: ctypes.windll.user32.BringWindowToTop(debugger_handler)
if TAKE_ROAD: ctypes.windll.user32.MoveWindow(game_handler, 0, 0, 272, 327, True)
if TAKE_ROAD: ctypes.windll.user32.BringWindowToTop(game_handler)

# -----------------
# Stream video data
# -----------------

from PIL import ImageGrab
import numpy as np
import cv2

road_stream = None
obj_stream = None

from PIL import ImageGrab
import numpy as np
import cv2

OBJ_STREAM_TOP_X = 851
OBJ_STREAM_TOP_Y= 60

ROAD_TOP_X = 8
ROAD_TOP_Y = 73

frame_count = 0

while(True):


    if TAKE_OBJECT: object = ImageGrab.grab(bbox=(OBJ_STREAM_TOP_X, OBJ_STREAM_TOP_Y, OBJ_STREAM_TOP_X + 256, OBJ_STREAM_TOP_Y + 112)) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)

    start = time.time()
    if TAKE_ROAD: road = ImageGrab.grab(bbox=(ROAD_TOP_X, ROAD_TOP_Y, ROAD_TOP_X + 256, ROAD_TOP_Y + 112)) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    end = time.time()
    print("Execution time = ", str(end - start))

    if TAKE_OBJECT: object = np.array(object)
    if TAKE_OBJECT: object = cv2.cvtColor(object, cv2.COLOR_BGR2GRAY)
    if TAKE_ROAD: road = np.array(road)
    if TAKE_ROAD: road = cv2.cvtColor(road, cv2.COLOR_BGR2RGB)

    if TAKE_OBJECT: object_topo = cv2.resize(object, (16, 7))
    if TAKE_OBJECT: object_topo = object_topo[:,:,0] > 0
    if TAKE_ROAD: road = cv2.resize(road, (16, 7))
    if TAKE_ROAD: road = road[:, :, 0] > 0



    # if TAKE_OBJECT:
    #     for line in object_topo:
    #         for element in line:
    #             if element == True:
    #                 print("1", end="")
    #             else:
    #                 print("0", end="")
    #         print()
    #
    # if TAKE_ROAD:
    #     for line in road:
    #         for element in line:
    #             if element == True:
    #                 print("1", end="")
    #             else:
    #                 print("0", end="")
    #         print()

    frame_count += 1
    if frame_count > FRAME_LIMIT: break


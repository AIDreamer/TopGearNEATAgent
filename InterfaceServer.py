from bottle import run, post, request, response, get, route
import time
import json
import numpy as np
import os

# ---------
# CONSTANTS
# ---------

# Game constants
BIZHAWK = True
IMG_SHAPE = (224, 256)
TOPO_SHAPE = (28, 32)
BUTTONS = ( "A","B","X","Y","Up","Down","Left","Right","L","R",)
ARRAY_LENGTH = TOPO_SHAPE[0] * TOPO_SHAPE[1] // 2
ARRAY_LENGTH = TOPO_SHAPE[0] * TOPO_SHAPE[1] // 2
ARRAY_LENGTH = TOPO_SHAPE[0] * TOPO_SHAPE[1] // 2
ARRAY_LENGTH = TOPO_SHAPE[0] * TOPO_SHAPE[1] // 2

# Coefficient
DIST_C = 1000
TIME_C = 1
RANK_C = 5000

# Base value for rotating variables
BASE_FITNESS = 5000 + 19 * RANK_C
BASE_DISTANCE = 88
BASE_TIME = 22520

# Base time and distance

# ---------
# UTILITIES
# ---------

def convert_int_arr_to_topo(int_arr, img_shape, topo_shape):

    # Reisze image
    int_arr = np.reshape(int_arr, img_shape)
    sh = topo_shape[0], int_arr.shape[0] // topo_shape[0], topo_shape[1], int_arr.shape[1] // topo_shape[1]
    int_arr = int_arr.reshape(sh).mean(-1).mean(1)

    # If it's not zero, it's part of topo
    return int_arr != 0

def topo_from_original_array(img_arr):

    # These are the integer form of the road color
    road_color1 = -12430766
    road_color2 = -14073278
    road_color3 = -6553600
    road_color4 = -3223858

    # Extract only those roads out.
    img_arr = np.array(img_arr)
    topo = np.any([img_arr == road_color1,img_arr == road_color2,img_arr == road_color3,img_arr == road_color4], axis=0)
    topo = convert_int_arr_to_topo(topo, IMG_SHAPE, TOPO_SHAPE)
    return topo

def print_key_presses(binary_key, key_names):
    s = ""
    for i in range(len(key_names)):
        if binary_key[i] == '0':
            s += " " * len(key_names[i])
        else:
            s += key_names[i]
        s += " "
    border = "-" * len(s)
    print(border + "\n" + s + "\n" + border)

def print_topo(topo):
    s = ""
    for i in range(topo.shape[0]):
        for j in range(topo.shape[1]):
            if topo[i][j]: s += "-"
            else: s += " "
        s += "\n"
    print(s)


# ----------------
# GLOBAL VARIABLES
# ----------------

key_press = "00000000000"
input_arr = np.zeros(ARRAY_LENGTH)

# -----------
# SERVER CODE
# -----------

@route('/cam_topo', method = 'POST')
def process_cam():
    global input_arr

    if (BIZHAWK):
        return("Finished")
    data = request.body.read().decode("utf-8")
    data = json.loads(data)
    main_topo = data["Topo"][1:-1]
    main_topo = "".join(main_topo.split("\n"))
    main_topo = np.fromstring(main_topo, sep=" ").reshape(TOPO_SHAPE[0] // 2, TOPO_SHAPE[1]).astype(int)

    os.system("cls")
    print_topo(main_topo)

    input_arr = main_topo.flatten()
    return("Finished")

@route('/screen', method = 'POST')
def process():
    global input_arr
    if (not BIZHAWK):
        time.sleep(0.01)
        return("Finished")


    # Get the data from the request
    start = time.time()
    data = request.body.read().decode('utf8')
    data = json.loads(data)
    pixels = data["Pixels"]
    rank = data["Rank"]
    speed = data["Speed"]

    # Determine a set of background color, we will try to remove this color from the game.
    road_topo = topo_from_original_array(pixels)[: TOPO_SHAPE[0] // 2, :].astype(int)
    end = time.time()

    # Print stuff
    os.system("cls")
    print_topo(road_topo)
    print_key_presses(key_press, BUTTONS)
    print("Speed:    " + str(speed))
    print("Rank:     " + str(rank))

    # Calculate decision from current net
    input_arr = road_topo.flatten()
    return("Finished")

@route('/control', method = 'GET')
def return_control():
    global key_press
    return key_press

@route('/get_input', method = 'GET')
def return_input():
    global input_arr
    topo_json = json.dumps({"Topo": np.array_str(input_arr)})
    return topo_json

@route('/post_control', method = 'POST')
def post_control():
    global key_press

    # Get data
    data = request.body.read().decode('utf8')
    data = json.loads(data)
    key_press = data["Keys"] + "0"

run(host='127.0.0.1', port=37979, debug=True)
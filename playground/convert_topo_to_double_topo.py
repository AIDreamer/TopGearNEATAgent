import numpy as np

def print_topo(topo):
    s = ""
    for i in range(topo.shape[0]):
        for j in range(topo.shape[1]):
            if topo[i][j]: s += "-"
            else: s += " "
        s += "\n"
    print(s)

def print_road_obj_topo(obj_topo, road_topo):
    s = ""
    for i in range(obj_topo.shape[0]):
        for j in range(obj_topo.shape[1]):
            if obj_topo[i][j]: s += "0"
            elif road_topo[i][j]: s+="-"
            else: s += " "
        s += "\n"
    print(s)

old_arr = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]])

new_arr = np.zeros(old_arr.shape)

for i in range(new_arr.shape[0]):
    meet_road = False
    for j in range(new_arr.shape[1]):
        if (old_arr[i][j]) == 1 and (meet_road == False):
            meet_road = True
        elif (old_arr[i][j] == 1) and (meet_road == True):
            fill_backwards(i, j - 1, old_arr, new_arr)
            meet_road = False

for j in range(new_arr.shape[1]):
    meet_road = False
    for i in range(new_arr.shape[0]):
        if (old_arr[i][j]) == 1 and (meet_road == False):
            meet_road = True
        elif (old_arr[i][j] == 1) and (meet_road == True):
            fill_upwards(i - 1, j, old_arr, new_arr)
            meet_road = False


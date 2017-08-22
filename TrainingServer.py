import subprocess
from bottle import run, post, request, response, get, route
import numpy as np
import cv2
import json
import codecs
import time
from PIL import Image
import os
import neat
from neat.six_util import iteritems, itervalues
import extract
import math

# ---------
# CONSTANTS
# ---------

# Game constants
IMG_SHAPE = (224, 256)
TOPO_SHAPE = (28, 32)
BUTTONS = ( "A","B","X","Y","Up","Down","Left","Right","L","R",)
CAR_BOX = (104, 71, 48, 28)

# Graphic constants
WIDTH = 256
HEIGHT = 224
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
ROAD_COLOR = (0.3, 0.3, 0.3, 1.0)
OBJ_COLOR = (0.0, 0.0, 0.0, 1.0)

SQUARE_UNIT_X = WIDTH // TOPO_SHAPE[1]
SQUARE_UNIT_Y = HEIGHT // TOPO_SHAPE[0]

# Neural net constant
NUM_INPUTS = TOPO_SHAPE[0] * TOPO_SHAPE[1]
THRESHOLD = 0.5
SANDSTALL_LIMIT = 50
STALL_LIMIT = 20
NUM_GENERATIONS = 50
NUM_GENOMES = 25

# Coefficient
DIST_C = 1000
TIME_C = 1
RANK_C = 5000

# Base value for rotating variables
BASE_FITNESS = 5000 + 19 * RANK_C
BASE_DISTANCE = 88
BASE_TIME = 22520

# Statistical variables
genome_rank = []
generation_rank = []
generation_rank_index = []

genome_fitness = [] # Storing all fitness value of current generation
generation_fitness = []
generation_fitness_index = []
generation_average_fitness = []

# Program states
STATE_BUFFERING = "BUFFERING"

STATE_BEGIN = "BEGIN"
STATE_EVALUATING = "EVALUATING"
STATE_PRESSING_F1 = "PRESSING F1"
STATE_RESETTED = "RESETED"
STATE_POST_PROCESSING = "POST-PROCESSING"
STATE_RETURN_BEST = "RETURNING THE BEST"
STATE_TEST_BEST = "TESTING THE FITTEST"

# ----------------
# CONTROL VARIABLE
# ----------------

# Because fitness has to be gathered along the way.
fitness_val = 0
key_press = "00000000000"

frame = 0
obj_topo = np.zeros(TOPO_SHAPE)
road_topo = np.zeros(TOPO_SHAPE)
program_state = STATE_BEGIN

new_distance = 0
old_distance = 0
base_distance = -BASE_DISTANCE
distance_stall = 0 # If distance is stall for a while. Then stop the training

new_time = 0
old_time = 0
base_time = -BASE_TIME

sand_stall = 0
x, y, w, h = CAR_BOX
RATIO_ROW = IMG_SHAPE[0] // TOPO_SHAPE[0]
RATIO_COL = IMG_SHAPE[1] // TOPO_SHAPE[1]
CAR_ROW_START = math.ceil(y / RATIO_ROW)
CAR_COL_START = math.ceil(x / RATIO_COL)
SHAPE_COL = math.floor(w / RATIO_COL)
SHAPE_ROW = math.floor(h / RATIO_ROW)

winner = None

# -----------------
# UTILITY FUNCTIONS
# -----------------

def convert_to_image(int_arr, shape = IMG_SHAPE):
    np_arr = np.array(int_arr)
    np_arr = np.reshape(np_arr, shape)
    np_arr = np.array(Image.fromarray(np_arr, "RGBA"))
    return np_arr

def convert_int_arr_to_topo(int_arr, img_shape, topo_shape):

    # Reisze image
    int_arr = np.reshape(int_arr, img_shape)
    sh = topo_shape[0], int_arr.shape[0] // topo_shape[0], topo_shape[1], int_arr.shape[1] // topo_shape[1]
    int_arr = int_arr.reshape(sh).mean(-1).mean(1)

    # If it's not zero, it's part of topo
    return int_arr != 0

def print_road_obj_topo(road_topo, obj_topo):
    s = ""
    for i in range(road_topo.shape[0]):
        for j in range(road_topo.shape[1]):
            if road_topo[i][j]: s += "0"
            elif obj_topo[i][j]: s+= "-"
            else: s += " "
        s += "\n"
    print(s)

def print_topo(topo):
    s = ""
    for i in range(topo.shape[0]):
        for j in range(topo.shape[1]):
            if topo[i][j]: s += "-"
            else: s += " "
        s += "\n"
    print(s)

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

def fill_backwards(row, col, old_arr, new_arr):
    for j in range(col, -1, -1):
        if old_arr[row][j] == 0:
            old_arr[row][j] = 1
            new_arr[row][j] = 1
        else: return

def fill_upwards(row, col, old_arr, new_arr):
    for i in range(row, -1, -1):
        if old_arr[i][col] == 0:
            new_arr[i][col] = 1
            old_arr[i][col] = 1
        else: return

def create_obj_topo(road_topo):
    obj_topo = np.zeros(road_topo.shape)

    for i in range(road_topo.shape[0]):
        meet_road = False
        meet_obj = False
        for j in range(road_topo.shape[1]):
            if (road_topo[i][j]) == 1 and (meet_road == False):
                meet_road = True
            elif (road_topo[i][j] == 0) and (meet_obj == False):
                meet_obj = True
            elif (road_topo[i][j] == 1) and (meet_road == True) and (meet_obj == True):
                fill_backwards(i, j - 1, road_topo, obj_topo)
                meet_obj = False

    # for j in range(road_topo.shape[1]):
    #     meet_road = False
    #     meet_obj = False
    #     for i in range(road_topo.shape[0]):
    #         if (road_topo[i][j]) == 1 and (meet_road == False):
    #             meet_road = True
    #         elif (road_topo[i][j] == 0) and (meet_obj == False):
    #             meet_obj = True
    #         elif (road_topo[i][j] == 1) and (meet_road == True) and (meet_obj == True):
    #             fill_upwards(i - 1, j, road_topo, obj_topo)
    #             meet_road = False

    # Fill in the car box
    obj_topo[CAR_ROW_START : CAR_ROW_START + SHAPE_ROW, CAR_COL_START : CAR_COL_START + SHAPE_COL] = 1

    return obj_topo

# ----------------
# HELPER FUNCTIONS
# ----------------

def reset_variables():
    global frame, distance_stall
    global sand_stall
    global old_distance, new_distance, base_distance
    global base_time, old_time, new_time
    global fitness_val

    distance_stall = 0  # If distance is stall for a while. Then stop the training
    old_distance = 0
    new_distance = 0
    base_distance = - BASE_DISTANCE

    base_time = -BASE_TIME
    old_time = 0
    new_time = 0

    sand_stall = 0

    fitness_val = 0
    frame = 0

# ------------------------
# NEURAL NETWORK UTILITIES
# ------------------------

def generate_key_presses(net, input_arr):
    # This will return a string of 10 boolean values that represent whether respective buttons are pressed
    # Order of buttons: A, B, X, Y, Up, Down, Left, Right, L, R
    output = net.activate(input_arr)
    s = ""
    for val in output:
        if val < THRESHOLD: s += "0"
        else: s += "1"
    return s

def convert_key_to_bin(input_arr):
    l = []
    for input in input_arr:
        if input == '0': l.append(0)
        else: l.append(1)
    return l

# Determine path to configuration file. This path manipulation is
# here so that the script will run successfully regardless of the
# current working directory.
local_dir = os.path.dirname(__file__)
config_file = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file)

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(1))

# Set up variable for training
current_net = None
generation_count = 0 # k
genome = None
genome_count = 0
genome_list = []
best_genome = None

# ----------
# WEB SERVER
# ----------

@route('/screen', method = 'POST')
def process():
    global program_state, p
    global config
    global obj_topo, road_topo
    global CAR_ROW_START, CAR_COL_START, CAR_SHAPE_COL, sand_stall
    global frame, new_distance, distance_stall, old_distance, base_distance
    global old_time, new_time, base_time
    global fitness_val, genome_fitness, generation_fitness, generation_fitness_index, generation_average_fitness
    global genome_rank, generation_rank, generation_rank_index
    global generation_count, genome_count, genome_list, genome, current_net, best_genome
    global key_press
    global winner

    print(program_state)

    # Stop the data processing if the agent is currently training
    if program_state == STATE_BUFFERING:
        time.sleep(0.1)
        pass

    elif program_state == STATE_BEGIN:
        program_state = STATE_BUFFERING

        # Report the current generation
        p.reporters.start_generation(p.generation)
        genome_list = list(iteritems(p.population))
        program_state = STATE_PRESSING_F1
        time.sleep(0.1)
        pass

    elif program_state == STATE_PRESSING_F1:
        time.sleep(0.1)
        pass

    elif program_state == STATE_RESETTED:
        program_state = STATE_BUFFERING

        # Grab the genome if there is still genome to check
        if genome_count < NUM_GENOMES:
            _, genome = genome_list[genome_count]
            current_net = neat.nn.FeedForwardNetwork.create(genome, config)
            program_state = STATE_EVALUATING
        else:
            genome_count = 0
            program_state = STATE_POST_PROCESSING
        pass

        # Sleep for the stability of the algorithm
        time.sleep(0.1)

    elif program_state == STATE_EVALUATING:

        old_distance = new_distance
        old_time = new_time

        # Get the data from the request
        start = time.time()
        data = request.body.read().decode('utf8')
        data = json.loads(data)

        pixels = data["Pixels"]
        speed = data["Speed"]
        new_time = data["Time"]
        if new_time > BASE_TIME: new_time = 0
        new_distance = data["Distance"]
        if new_distance == BASE_DISTANCE: new_distance = 0
        rank = data["Rank"]
        nitro = data["Nitro"]

        # Determine a set of background color, we will try to remove this color from the game.
        road_topo = topo_from_original_array(pixels)[: TOPO_SHAPE[0] // 2, :]
        end = time.time()

        # Calculate decision from current net
        input_arr = road_topo.flatten()
        key_press = generate_key_presses(current_net, input_arr) + "0" # Don't press F1 to reset

        # Update frame counter and distance_stall and time
        if new_distance < old_distance: base_distance += BASE_DISTANCE
        if new_distance == old_distance:
            distance_stall += 1
        else:
            distance_stall = 0
        frame += 1
        if new_time < old_time: base_time += BASE_TIME
        if not road_topo[CAR_ROW_START + SHAPE_ROW, CAR_COL_START: CAR_COL_START + SHAPE_COL].any():
            sand_stall += 1
        else:
            sand_stall = 0

        fitness_val = (base_distance + new_distance) * DIST_C \
                      - (base_time + new_time) * TIME_C \
                      - rank * RANK_C \
                      + BASE_FITNESS

        # Print necessary information
        os.system('cls')
        print("Current generation: " + str(generation_count) + "/" + str(NUM_GENERATIONS))
        print("----")
        print_topo(road_topo)
        print(end - start)
        print_key_presses(key_press, BUTTONS)
        print("Speed:    " + str(speed))
        print("Time:     " + str(base_time + new_time))
        print("Distance: " + str(base_distance + new_distance))
        print("Rank:     " + str(rank))
        print("Nitro:    " + str(nitro))
        print("Current genome: " + str(genome_count + 1) + "/" + str(NUM_GENOMES))
        print("Fitness value: " + str(fitness_val))
        print('--------------')
        print('| Gen | Best Agent | Score      | Rank | Avg Score   |')
        print('|----------------------------------------------------|')
        for i in range(len(generation_fitness)):
            print("| {:3d} | {:10d} | {:10d} | {:4d} | {:11.3f} |".format( \
                i, generation_fitness_index[i], generation_fitness[i], generation_rank[i], generation_average_fitness[i]))

        # Stop the running if stall limit is reached
        if distance_stall > STALL_LIMIT:
            genome.fitness = fitness_val
            print("- Evaluated Genome " + str(genome_count) + ": " + str(fitness_val))

            # Add a bunch of statistical variables
            genome_fitness.append(fitness_val)
            genome_rank.append(rank)

            # Reset
            reset_variables()
            genome_count += 1
            program_state = STATE_PRESSING_F1

        # Return something cuz a post go with a get
        pass

    elif program_state == STATE_POST_PROCESSING:

        program_state = STATE_BUFFERING

        # Gather and report statistics.
        best = None
        for g in itervalues(p.population):
            if best is None or g.fitness > best.fitness:
                best = g
        p.reporters.post_evaluate(p.config, p.population, p.species, best)

        # Track the best gen ome ever seen.
        if p.best_genome is None or best.fitness > p.best_genome.fitness:
            p.best_genome = best

        # End if the fitness threshold is reached.
        fv = p.fitness_criterion(g.fitness for g in itervalues(p.population))
        if fv >= p.config.fitness_threshold:
            p.reporters.found_solution(p.config, p.generation, best)
            program_state = STATE_RETURN_BEST

        # Create the next generation from the current generation.
        p.population = p.reproduction.reproduce(p.config, p.species,
                                                p.config.pop_size, p.generation)

        # Check for complete extinction.
        if not p.species.species:
            p.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            # otherwise raise an exception.

            if p.config.reset_on_extinction:
                p.population = p.reproduction.create_new(p.config.genome_type,
                                                         p.config.genome_config,
                                                         p.config.pop_size)
            else:
                raise neat.CompleteExtinctionException()

        # Divide the new population into species.
        p.species.speciate(p.config, p.population, p.generation)
        p.reporters.end_generation(p.config, p.population, p.species)
        p.generation += 1

        # Add the best genome value into the statistics
        generation_fitness.append(max(genome_fitness))
        generation_average_fitness.append(np.average(genome_fitness))
        generation_fitness_index.append(np.argmax(genome_fitness))
        genome_fitness = []

        generation_rank.append(min(genome_rank))
        generation_rank_index.append(np.argmin(genome_rank))
        genome_rank = []

        # Increment generation count and switch state
        np.savetxt("best_genome_by_generation.npy",generation_fitness_index)
        generation_count += 1
        if generation_count == NUM_GENERATIONS: program_state = STATE_RETURN_BEST
        else: program_state = STATE_BEGIN
        pass

    elif program_state == STATE_RETURN_BEST:
        best_genome = p.best_genome
        program_state = STATE_TEST_BEST
        pass

    elif program_state == STATE_TEST_BEST:
        print("Should start testing stuff now")
        pass

    return("Finished")

@route('/control', method = 'GET')
def returnControl():
    global key_press, program_state

    print(program_state)
    # Order of buttons: A, B, X, Y, Up, Down, Left, Right, L, R, F11
    if program_state == STATE_PRESSING_F1:
        key_press = "00000000000" # Reset keypress to avoid future conflicting key pressed
        program_state = STATE_RESETTED
        print("THE BUTTON IS FUCKING PRESSED")
        return("00000000001") # Press F1 to reset
    elif program_state == STATE_EVALUATING:
        return key_press
    else:
        return ("00000000000")

run(host='127.0.0.1', port=37979, debug=True)
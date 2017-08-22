import neat
from neat.six_util import iteritems
import numpy as np
import requests
import json
import time

# --------
# CONSTANT
# --------

CHOSEN_GEN = 10
CHOSEN_GENOME = 1
TOPO_SHAPE = (28, 32)
THRESHOLD = 0.5

# ---------
# UTILITIES
# ---------

def generate_key_presses(net, input_arr):
    # This will return a string of 10 boolean values that represent whether respective buttons are pressed
    # Order of buttons: A, B, X, Y, Up, Down, Left, Right, L, R
    output = net.activate(input_arr)
    s = ""
    for val in output:
        if val < THRESHOLD: s += "0"
        else: s += "1"
    return s

# ---------------
# PREPARE NETWORK
# ---------------

# best_agents = np.load("./data/from_tung/best_genome_by_generation.npy")
p = neat.Checkpointer.restore_checkpoint('./data/from_tung/neat-checkpoint-' + str(CHOSEN_GEN))
genome_list = list(iteritems(p.population))
_, genome = genome_list[CHOSEN_GENOME]
current_net = neat.nn.FeedForwardNetwork.create(genome, p.config)

# --------------
# NEURAL NETWORK
# --------------

while (True):
    r = requests.get("http://127.0.0.1:37979/get_input")
    data = r.json()
    main_topo = data["Topo"][1:-1]
    main_topo = "".join(main_topo.split("\n"))
    main_topo = np.fromstring(main_topo, sep=" ")

    key_press = generate_key_presses(current_net, main_topo)
    requests.post("http://127.0.0.1:37979/post_control", data = json.dumps({"Keys": key_press}))
    time.sleep(1 / 60)
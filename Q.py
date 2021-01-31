import numpy as np
from config import config

def initialize_Qtable(sizes):
    qt = np.zeros(config.q_table_size_array + [2])
    return qt

def val_to_coord(val, val_min_max, nb_steps):
    val_min = val_min_max[0]
    val_max = val_min_max[1]

    if (val <= val_min):
        return (0)
    if (val >= val_max):
        return (nb_steps - 1)

    val = config.transfo(val)
    val_min = config.transfo(val_min)
    val_max = config.transfo(val_max)
    
    val_adjusted = val - val_min
    span = val_max - val_min
    step = span / nb_steps
    n = val_adjusted / step
    return int(n)


def state_to_q_table_coord(state, q_table_size = config.q_table_size_array, state_limits = config.state_space_limits):
    out = []
    for i in range(len(state)):
        out.append(val_to_coord(state[i], state_limits[i], q_table_size[i]))
        print(out)
    return (out)

if __name__ == "__main__":
    print("Tests for " + "state_to_q_table_coord")
    states = [[0, 0, 0, 0],
              [1, 1, 1, 1], 
              [-1, -1, -1, -1], 
              [0.5, 0.5, 0.5, 0.5],
              [-0.5, -0.5, -0.5, -0.5],
              [50, 50, 50, 50],
              [-50, -50, -50, -50],
              [1, 0, 1, 0]]
    for s in states:
        print("In:  ", s)
        o = state_to_q_table_coord(s)
        print("Out: ", o)
        print("")
    


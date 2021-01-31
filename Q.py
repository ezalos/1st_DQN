import numpy as np
from config import config
import random
import matplotlib.pyplot as plt


def initialize_Qtable(sizes = config.qt_size_array):
    qt = np.zeros(config.qt_size_array + [2])
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


def state_to_qt_coord(state,
            qt_size = config.qt_size_array,
            state_limits = config.state_space_limits):
    out = []
    for i in range(len(state)):
        out.append(val_to_coord(state[i], state_limits[i], qt_size[i]))
    return (tuple(out))

def choose_action(state, qt, epsilon = config.epsilon):
    coords = state_to_qt_coord(state)
    if (max(qt[coords]) == 0):
        return random.randint(0,1)
    if (random.random() > epsilon):
        return np.argmax(qt[tuple(coords)])
    else:
        return random.randint(0,1)


def choose_greedy_action(state, qt, epsilon = config.epsilon):
    coords = state_to_qt_coord(state)
    if (max(qt[coords]) == 0):
        return random.randint(0,1)
    else:
        return np.argmax(qt[coords])
        


def bellman_q(state, qt, cart, depth = config.future_steps, action = None,
            learning_rate = config.learning_rate,
            discount_factor = config.discount_factor):
    '''
    negative reward is not yet set to electroshock
    '''
    if (action == None):
        action = choose_greedy_action(state, qt)
    new_state, reward, terminal, _ = cart.step(action)
    # reward = config.reward_values.get(int(reward), 0)
    print("rewa:" , reward)
    q = reward
    if (depth != 0 and not terminal):
        return q + discount_factor * (bellman_q(new_state, qt, cart, depth - 1))
    else:
        # return (q + (discount_factor * max(qt[state_to_qt_coord(state)])))
        return q


def update_qt(qt, state, action, temporal_difference_target, learning_rate = config.learning_rate):
    coords = state_to_qt_coord(state)
    old_value = qt[coords][action]
    temporal_difference = temporal_difference_target - old_value
    qt[coords][action] = (1 - learning_rate) * old_value + learning_rate * temporal_difference


def dummy_cart(s, cart = None):
    if cart == None:
        cart = CartPoleEnv()
    cart.reset()
    cart.state = s
    return cart
    

def graph(data, rm):
    
    plt.scatter(list(range(len(data))), data, s=1, c="b")
    plt.plot(rm, c="r")
    # plt.autoscale_view()
    plt.pause(0.05)
    plt.show()

import numpy as np

def loop(qt = None, epsilon = 1, visu = False):
    plt.ion()
    cart = CartPoleEnv()
    data = []
    data_rm = []
    config.epsilon = epsilon
    if (qt is None):
        qt = initialize_Qtable()
    for episode in range(config.episodes):
        cart.reset()
        turn = 0
        s = cart.state
        end = False
        epsilon_tmp = config.epsilon
        while not end:
            config.epsilon *= 0.97
            if (visu):
                cart.render()
            a = choose_action(s, qt)
            _, _, end, _ = cart.step(a)
            l_val = bellman_q(s, qt, dummy_cart(s), action = a)
            print(l_val)
            update_qt(qt, s, a, l_val)
            s = cart.state
            turn += 1
        data.append(turn)
        data_rm.append(np.mean(data[-100:]))
        print("Episode: ", episode, "\tTurn:", turn, "\t Epsilon:", config.epsilon)
        config.epsilon = epsilon_tmp
        if episode % config.graph_update == 0 and episode != 0:
            graph(data, data_rm)
        # if ((episode + 1) % 100 == 0 and input("continue (y/n)" != "y")):
        #     break
    cart.close()
    return (data, qt)

from cartpole import CartPoleEnv

if __name__ == "__main__":
    data, qt = loop()


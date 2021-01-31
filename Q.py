MIN_INF = -10000.0
MAX_INF =  10000.0

def trqnsfo_sqrt(window, value, slots):
    transfo = lambda x: (abs(x) ** 0.5) * (x / abs(x))
    min = transfo(window[0])


def initialize_Qtable():
    c_pos = [-4.8, 4.8]
    c_vel = [MIN_INF, MAX_INF]
    a_ang = [-0.418, 0.418]
    a_vel = [MIN_INF, MAX_INF]

    ranges = [c_pos, c_vel, a_ang, a_vel]

    size = 4 ** 4



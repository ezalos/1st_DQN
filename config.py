from utils import DotDict
from functools import reduce

config = DotDict()


config.state_space_limits = [[-4.8, 4.8], [-20, 20], [-0.418, 0.418], [-10, 10]] #[Cart Pos, Cart Velocity, Pole Angle, Pole Angular Velocity]

config.q_table_size_array = [4,4,5,5] # [Cart Pos, Cart Velocity, Pole Angle, Pole Angular Velocity]
config.q_table_size = reduce((lambda x, y: x * y), config.q_table_size_array)
config.discretisation_method = "linear"  # sqrt, linear
config.electroshock = -1000 #negative reward

if (config.discretisation_method == "sqrt"):
    config.transfo = lambda x: (abs(x) ** 0.5) * (x / abs(x)) if x != 0 else 0
else:
    config.transfo = lambda x: x
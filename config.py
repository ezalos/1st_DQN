from utils import DotDict
from functools import reduce

config = DotDict()


# config.state_space_limits = [[-4.8, 4.8], [-20, 20], [-0.418, 0.418], [-10, 10]] #[Cart Pos, Cart Velocity, Pole Angle, Pole Angular Velocity]
config.state_space_limits = [[-4.8 / 1.5, 4.8 / 1.5] , [-2, 2], [-0.418 / 1.5, 0.418 / 1.5], [-4, 4]] #[Cart Pos, Cart Velocity, Pole Angle, Pole Angular Velocity]

config.qt_size_array = [3,3,6,6] # [Cart Pos, Cart Velocity, Pole Angle, Pole Angular Velocity]
config.qt_size = reduce((lambda x, y: x * y), config.qt_size_array)
config.discretisation_method = "sqrt"  # sqrt, linear
config.reward_values = {0 : -10, 1 : 1}
config.future_steps = 5
config.discount_factor = 0.9
config.learning_rate = 0.7
config.epsilon = 0.9
config.episodes = 100000

if (config.discretisation_method == "sqrt"):
    config.transfo = lambda x: (abs(x) ** 0.5) * (x / abs(x)) if x != 0 else 0
else:
    config.transfo = lambda x: x

config.graph_update = 2000
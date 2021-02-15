import numpy as np
from ModelsManager import ModelsManager
from utils import DotDict
from functools import reduce

config = DotDict()


# config.state_space_limits = [[-4.8, 4.8], [-20, 20], [-0.418, 0.418], [-10, 10]] #[Cart Pos, Cart Velocity, Pole Angle, Pole Angular Velocity]
config.state_space_limits = [[-4.8 / 1.5, 4.8 / 1.5] , [-2, 2], [-0.418 / 1.5, 0.418 / 1.5], [-4, 4]] #[Cart Pos, Cart Velocity, Pole Angle, Pole Angular Velocity]

config.qt_size_array = [11,11,11,11] # [Cart Pos, Cart Velocity, Pole Angle, Pole Angular Velocity]
config.qt_size = reduce((lambda x, y: x * y), config.qt_size_array)
config.discretisation_method = "sqrt"  # sqrt, linear
config.reward_values = {0 : -10, 1 : 1}
config.future_steps = 5
config.discount_factor = 0.9
config.learning_rate = 0.02
config.epsilon = 0.9
config.episodes = 100_000

if (config.discretisation_method == "sqrt"):
    config.transfo = lambda x: (abs(x) ** 0.5) * (x / abs(x)) if x != 0 else 0
else:
    config.transfo = lambda x: x

config.graph_update = 200
config.graph_running_mean = 100
config.graph_window = True

net_config = DotDict()
GridS_conf = DotDict()

# MODEL Hyper Parameters

# GridSearchCV Params:
net_config.layers = [4, 64, 128, 2]
GridS_conf.gsc_layers = [[4, 64, 128, 2],
                         [4, 8, 16, 32, 2],
                         [4, 8, 16, 32, 64, 2],
                         [4, 16, 256, 2],
                         [4, 6, 8, 4, 2]]
net_config.learning_rate = 0.005
GridS_conf.gsc_learning_rate = [0.0005, 0.001, 0.005, 0.01]
net_config.gamma = 0.9
GridS_conf.gsc_gamma = np.linspace(0.5, 1, 5)
net_config.n_update = 25
GridS_conf.gsc_n_update = np.linspace(10, 200, 5)
net_config.epsilon = 0.9
GridS_conf.gsc_epsilon = np.linspace(0.5, 1, 5)
net_config.min_epsilon = 0.01
GridS_conf.gsc_min_epsilon = np.linspace(0.01, 0.3, 5)
net_config.eps_decay = 0.995
GridS_conf.gsc_eps_decay = [0.99, 0.995, 0.999]
net_config.batch = 100
GridS_conf.gsc_batch = np.linspace(10, 200, 10)
net_config.replay_nb_batch = 2
GridS_conf.gsc_replay_nb_batch = np.linspace(0, 10, 5)
net_config.reward_optimisation = False
GridS_conf.gsc_reward_optimisation = [True, False]
net_config.tau = 0.1
GridS_conf.gsc_tau = np.linspace(0.1, 0.5, 5)
net_config.soft_update = True
GridS_conf.gsc_soft_update = [True, False]
net_config.max_turns = 500
GridS_conf.gsc_max_turns = np.linspace(200, 1000, 5)
net_config.dropout = 0.1
GridS_conf.gsc_dropout = np.linspace(0, 0.5, 5)
net_config.early_stopping = False
GridS_conf.gsc_early_stopping = [True, False]

net_config.ModelsManager = ModelsManager()
# GridS_conf.gsc_ModelsManager = [ModelsManager()]

#TODO add   min_epsilon
#           ModelsManager

# Unused
net_config.reward_loose = 0
net_config.reward_win = 1
net_config.memory_size = 100_000

net_config.seed = 42

# Environnement Parameters
net_config.max_turns = 500
net_config.max_episodes = 1500
net_config.consecutive_wins_required = 100
net_config.turn_threshold_to_win = 195

net_config.eval_episodes = 150

user_config = DotDict()
user_config.visu = False
user_config.visu_update = False  # 300
user_config.visu_window = 5
user_config.plot = False
user_config.verbose = 2


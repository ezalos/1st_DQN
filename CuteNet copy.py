import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

from config import net_config

dico = OrderedDict()


netlist = []
n = 0
for i, o in zip(net_config.layers[:-1], net_config.layers[1:]):
    if (n != 0):
        netlist.append(nn.LeakyReLU())
    netlist.append(nn.Linear(i, o))
    n += 1

class DQN():
    def __init__(self, layers = net_config.layers):
        self.model = nn.Sequential(netlist)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), net_config.learning_rate)

    def update(self, state, y):
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        '''
        Why torch no grad ?
        '''
        with torch.no_grad():
            return self.model(torch.Tensor(state))


def q_learning_loopidoop():
    main_net = DQN()
    second_net = deepcopy(main_net)
    while (training):
        train_data : [state, second_net_prediction(state)]
        main_net.update(state, second_net_prediction(state))
        if n % 10 = 0
            copy_parmams_to_second_net()


def learn_ma_boy(env, main_net, target_net, n_update = net_config.n_update, gamma = config.discount_factor, epsilon = config.epsilon, eps_decay = 0.99, memory)

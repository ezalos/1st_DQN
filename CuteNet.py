import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

from config import net_config
from copy import deepcopy
import random
import numpy as np
from config import config

from cartpole import CartPoleEnv


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
        self.model = nn.Sequential(* netlist)
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

def choose_action_net(y_pred, epsilon = config.epsilon):
    if (random.random() > epsilon):
        return int(np.argmax(y_pred))
    else:
        return random.randint(0,1)

import matplotlib.pyplot as plt

def graph(data, rm):
    if False:
        plt.scatter(list(range(len(data))), data, s=1, c="b")
    plt.plot(rm, c="r")
    # plt.autoscale_view()
    plt.pause(0.05)
    plt.show()

def q_learning_loopidoop():
    plt.ion()
    cart = CartPoleEnv()
    cart.reset()
    predi_net = DQN()
    updat_net = deepcopy(predi_net)
    data = []
    data_rm = []
    turn = 0
    episode = 0
    while True:
        s = cart.state
        y = predi_net.predict(s)
        a = choose_action_net(y)
        _, _, end, _ = cart.step(a)
        y[a] = -10 if end else 1
        updat_net.update(s, y)
        if n % net_config.n_update == 0 and n:
            predi_net = deepcopy(updat_net)
        turn += 1
        if end:
            episode +=1
            data.append(turn)
            data_rm.append(np.mean(data[-100:]))
            print("Episode: ", episode, "\tTurn:", turn, "\t Epsilon:", config.epsilon)
            turn = 0
            cart.reset()
            if episode % config.graph_update == 0 and episode != 0:
                graph(data, data_rm)
    cart.close()

if __name__ == "__main__":
    q_learning_loopidoop()

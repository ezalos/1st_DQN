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


def graph(data, rm):
    if False:
        plt.scatter(list(range(len(data))), data, s=1, c="b")
    plt.plot(rm, c="r")
    # plt.autoscale_view()
    plt.pause(0.05)
    plt.show()

from plot_data import PlotData

class CuteLearning():
    def __init__(self):
        self.plot_data = PlotData()
        self.cart = CartPoleEnv()
        self.cart.reset()
        self.predi_net = DQN()
        self.updat_net = deepcopy(self.predi_net)
        self.turn = 0
        self.epidode = 0

    def learn(self):
        self.turn = 0
        self.episode = 0
        n = 0
        while True:
            state = self.cart.state
            y = self.predi_net.predict(state)
            a = choose_action_net(y)
            next_state, _, end, _ = self.cart.step(a)
            reward = -10 if end else 1

            q_values_next = self.predi_net.predict(next_state)
            y[a] = reward + net_config.gamma * torch.max(q_values_next).item()
            self.updat_net.update(state, y)
            self.turn += 1

            if n % net_config.n_update == 0 and n:
                self.predi_net = deepcopy(self.updat_net)
            if end:
                self.end()

            n += 1
        self.cart.close()
        self.plot_data.clear()

    def end(self):
        self.episode += 1
        self.plot_data.new_data(self.turn)
        print("Episode: ", self.episode, "\tTurn:", self.turn, "\t Epsilon:", config.epsilon)
        self.turn = 0
        self.cart.reset()
        if self.episode % config.graph_update == 0 and self.episode != 0:
            self.plot_data.graph()

if __name__ == "__main__":
    Cutie = CuteLearning()
    Cutie.learn()

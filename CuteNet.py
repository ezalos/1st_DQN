from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

import matplotlib.pyplot as plt
from config import net_config
from copy import deepcopy
import random
import numpy as np
from config import config

from cartpole import CartPoleEnv

import pickle
from datetime import datetime

import json

dico = OrderedDict()

netlist = []
n = 0
for i, o in zip(net_config.layers[:-1], net_config.layers[1:]):
    print("Layer ", n, "[", i, ":", o, "]")
    if (n != 0):
        netlist.append(nn.LeakyReLU())
    netlist.append(nn.Linear(i, o))
    if n != 0 and n != len(net_config.layers[:-1]) - 1:
        print("Dropout")
        # netlist.append(nn.Dropout(net_config.dropout))
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
        self.epsilon = config.epsilon
        self.eps_decay = 0.999
        self.visu = False
        self.visu_update = 0
        self.visu_window = 5
        self.consecutive_wins = 0
        self.best_consecutive_wins = 0
        self.last_save = 0

    def learn(self):
        self.turn = 0
        self.episode = 0
        n = 0
        while True:
            state = self.cart.state
            y = self.predi_net.predict(state)
            a = choose_action_net(y, self.epsilon)
            next_state, _, end, _ = self.cart.step(a)
            reward = -25 if end else 1
            if reward == 1:
                reward += (((abs((0.418 / 2) - abs(next_state[2])) / (0.418 / 2)) * 2) - 1) * 2
                reward += (((abs((4.8 / 2) - abs(next_state[0])) / 2.4) * 2) - 1) * 2
                #reward += abs((2.4) - abs(state[0])) * .6
            q_values_next = self.predi_net.predict(next_state)
            y[a] = reward + net_config.gamma * torch.max(q_values_next).item()
            self.updat_net.update(state, y)
            self.turn += 1
            if self.visu:
                self.cart.render()
            if n % net_config.n_update == 0 and n:
                self.predi_net = deepcopy(self.updat_net)
            if self.turn >= 500:
                end = True
            if end:
                self.end()
            n += 1

        self.cart.close()
        self.plot_data.clear()

    def end(self):
        self.plot_data.new_data(self.turn)
        if self.turn > 195:
            self.consecutive_wins += 1
            if self.best_consecutive_wins < self.consecutive_wins:
                self.best_consecutive_wins = self.consecutive_wins
            if self.consecutive_wins > 200:
                print(("WIN IN " + str(self.episode) + " EPISODES\n") * 100)
        else:
            self.consecutive_wins = 0
            if self.last_save * 1.2 < self.best_consecutive_wins and 50 <= self.best_consecutive_wins:
                self.save()
                self.last_save = self.best_consecutive_wins
        print("Episode: ", self.episode, "\tTurn:",
              self.turn, "\tEpsilon:", self.epsilon,
              "\tWins: ", self.consecutive_wins)
        self.turn = 0
        self.cart.reset()
        if self.episode % config.graph_update == 0 and self.episode != 0:
            self.plot_data.graph()
        if self.visu_update:
            if self.episode % self.visu_update == 0:
                self.visu = True
            if self.episode % self.visu_update == self.visu_window:
                self.visu = False
                self.cart.close()
        self.epsilon = max(self.epsilon * self.eps_decay, 0.01)
        self.episode += 1

    def save(self):
        name = "model_cache/"
        name += str(self.best_consecutive_wins) + "Wins"
        name += "_"
        name += str(self.epidode) + "Episodes"
        name += "_"
        now = datetime.now()
        name += now.strftime("%d-%m %H:%M")
        with open(name + ".mdl", "wb+") as f:
            pickle.dump(self, f)
        self.plot_data.save(name)
        with open(name + ".json", "w+") as f:
            json.dump(net_config, f, indent=4)

if __name__ == "__main__":
    Cutie = CuteLearning()
    Cutie.learn()

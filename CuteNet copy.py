from torch.autograd import Variable
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
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
from dataloader import Memory

netlist = []
n = 0
for i, o in zip(net_config.layers[:-1], net_config.layers[1:]):
    print("Layer ", n, "[", i, ":", o, "]")
    if (n != 0):
        netlist.append(nn.LeakyReLU())
    netlist.append(nn.Linear(i, o))
    #dropout
    n += 1


class DQN():
    def __init__(self, layers = net_config.layers):
        self.model = nn.Sequential(* netlist)
        self.criterion = nn.MSELoss(reduction = "mean")
        self.optimizer = torch.optim.Adam(self.model.parameters(), net_config.learning_rate)

    def update(self, state, y):
        y_pred = self.model(torch.Tensor(state))
        logging.debug(f"state: {state}")
        logging.debug(f"torch tensor state {torch.Tensor(state)}")
        logging.debug(f"y_pred {y_pred}")
        logging.debug(f"y {y}")
        logging.debug("\n\n")
        loss = self.criterion(y_pred, torch.Tensor(y))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        '''
        Why torch no grad ?
        '''
        with torch.no_grad():
            return self.model(torch.Tensor(state))

# def choose_action_net(state, epsilon = config.epsilon):
#     if (random.random() > epsilon):
#         return int(np.argmax(y_pred))
#     else:
#         return random.randint(0,1)


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
        self.env = CartPoleEnv()
        self.main_net = DQN()
        self.target_net = deepcopy(self.main_net)
        self.epsilon = config.epsilon
        self.eps_decay = 0.995
        self.visu = False
        self.visu_update = False#300
        self.visu_window = 5
        self.memory = Memory(memory_size = 30)
        self.batch_size = 5

    def reward_optimisation(self, state, end):
        reward = 0 if end else 1
        return reward
    
    def choose_action(self, q_values):
        if (random.random() > self.epsilon):
            return(np.argmax(q_values))
        else:
            return random.randint(0,1)

    def make_batch(self):
        batch = self.memory.get_batch(self.batch_size)
        states = []
        targets = []
        for s, a, r, ns, done in batch:
            states.append(s)
            q_values = self.target_net.predict(s).tolist()
            if done:
                q_values[a] = r
            else:
                q_values_next = self.target_net.predict(ns)
                q_values[a] = r + net_config.gamma * torch.max(q_values_next).item()
            targets.append(q_values)
        return states, targets

    def updato(self):
        states, targets = self.make_batch()
        self.main_net.update(states, targets)


    def learn(self, episodes = 10000, replay = False):
        episode = 0
        tmp = self.epsilon
        while (episode < episodes):
            done = False
            turn = 0
            state = self.env.reset()
            self.eps_decay = self.epsilon * self.eps_decay
            while (done == False):
                q_values = self.main_net.model(torch.Tensor(state)).tolist()
                action = self.choose_action(q_values)
                new_state, reward, done, _ = self.env.step(action)
                self.memory.add_data((state, action, reward, new_state, done))
                state = new_state
                turn += 1
                self.updato()


            print("turn:", turn)
            episode += 1
            if (episode % net_config.n_update == 0):
                self.target_net = deepcopy(self.main_net)
        self.epsilon = tmp
    def save(self):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Cutie = CuteLearning()
    Cutie.learn()

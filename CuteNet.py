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
        logging.info("state", state)
        logging.info("torch tensor state", torch.Tensor(state))
        logging.info("y_pred", y_pred)
        logging.info("y", y)
        logging.info("\n\n")
        loss = 0
        # for i in range(len(y)):
        #     loss += self.criterion(y_pred[i], y[i])
        # print(y_pred, y)
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
        self.eps_decay = 0.99
        self.visu = False
        self.visu_update = False#300
        self.visu_window = 5
        self.consecutive_wins = 0
        self.best_consecutive_wins = 0
        self.last_save = 0
        self.memory = []

    def reward_optimisation(self, state, end):
        reward = -25 if end else 1
        if reward == 1:
            # Angle reward modification
            angle_r = 0.418 / 2
            reward += (((abs(angle_r - abs(state[2]))
                        / angle_r) * 2) - 1) * 2
            # Position reward modification
            pos_r = 0.418 / 2
            reward += (((abs(pos_r - abs(state[0]))
                        / pos_r) * 2) - 1) * 2
        return reward
    
    def learn(self):
        self.episode = 0
        n = 0
        while self.episode < 10:
            self.turn = 0
            end = False
            states = []
            targets = []
            while not end:
                # 1. Init
                state = self.cart.state
                # 2. Choose action
                q_values = self.predi_net.predict(state).tolist()
                a = choose_action_net(q_values, self.epsilon)
                # 3. Perform action
                next_state, _, end, _ = self.cart.step(a)
                # 4. Measure reward
                reward = self.reward_optimisation(next_state, end)
                q_values_next = self.predi_net.predict(next_state)
                # 5. Calcul Q-Values
                q_values[a] = reward + net_config.gamma * \
                    torch.max(q_values_next).item()

                self.turn += 1
                self.memory.append((state, a, next_state, reward, end))
                # self.updat_net.update(state, q_values)
                states.append(state)
                targets.append(q_values)
                if (self.turn % 20 and self.turn) or end:
                    self.updat_net.update(states, targets)
                    states = []
                    targets = []

                if self.turn >= 500:
                    end = True
                if self.visu:
                    self.cart.render()

            self.episode += 1
            self.replay(20)
            if self.episode % net_config.n_update == 0 and self.episode:
                print("Update")
                self.predi_net.model.load_state_dict(self.updat_net.model.state_dict())
            self.end()
            n += 1
        
        self.save()
        self.cart.close()
        self.plot_data.clear()

    def replay(self, size):
        if size > len(self.memory):
            size = len(self.memory)
        data = random.sample(self.memory, size)
        states = []
        targets = []
        for state, action, next_state, reward, done in data:
            q_values = self.predi_net.predict(state)
            if done:
                q_values[action] = reward
            else:
                # The only difference between the simple replay is in this line
                # It ensures that next q values are predicted with the target network.
                q_values_next = self.predi_net.predict(next_state)
                q_values[action] = reward + net_config.gamma * torch.max(q_values_next).item()
            states.append(state)
            targets.append(q_values)
        self.updat_net.update(state, q_values)


    def end(self):
        self.plot_data.new_data(self.turn)
        if self.turn > 195:
            self.consecutive_wins += 1
            if self.best_consecutive_wins < self.consecutive_wins:
                self.best_consecutive_wins = self.consecutive_wins
            if self.consecutive_wins > 200:
                self.save()
                print(("WIN IN " + str(self.episode) + " EPISODES\n") * 100)
        else:
            self.consecutive_wins = 0
            if self.last_save * 1.2 < self.best_consecutive_wins and 50 <= self.best_consecutive_wins:
                self.save()
                self.last_save = self.best_consecutive_wins
        print("Episode: ", self.episode, "\tTurn:",
              self.turn, "\tEpsilon:", self.epsilon,
              "\tWins: ", "{:3}".format(self.consecutive_wins),
              "/", self.best_consecutive_wins)
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

    def save(self):
        pass
        # name = "model_cache/"
        # name += str(self.best_consecutive_wins) + "Wins"
        # name += "_"
        # name += str(self.episode) + "Episodes"
        # name += "_"
        # now = datetime.now()
        # name += now.strftime("%d-%m %H:%M")
        # with open(name + ".mdl", "wb+") as f:
        #     pickle.dump(self, f)
        # self.plot_data.save(name)
        # with open(name + ".json", "w+") as f:
        #     json.dump([config, net_config], f, indent=4,
        #               default=lambda o: '<not serializable>')

if __name__ == "__main__":
    Cutie = CuteLearning()
    Cutie.learn()

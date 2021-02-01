import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

from config import net_config, config
import numpy as np
import random
import copy


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


def choose_action(q_values, epsilon = config.epsilon):
    if (random.random() > epsilon):
        return int(np.argmax(q_values))
    else:
        return random.randint(0,1)

def learn_from_memory(net:DQN, memory):
    for state, action, q_values, new_state, reward in memory:
        net.update(state, q_values)


def learn_ma_boy(env, main_net, target_net, memory_size = 20,
    n_update = net_config.n_update, gamma = config.discount_factor, epsilon = config.epsilon, eps_decay = 0.99, episodes = config.episodes):

    memory = []
    for episode in episodes:
        state, reward, done, _ = env.reset()
        total = 0
        while not done:
            q_values = target_net.predict(state).item()
            action = choose_action(q_values)
            new_state, reward, done, _ = env.step(action)
            q_values[action] = reward + gamma * target_net.predict(state).item()[action]
            memory.append(state, action, q_values, new_state, reward)
            state = new_state
            total += reward
        if (episode != 0 and episode % n_update == 0):
            learn_from_memory(main_net, memory)
            target_net = copy.deepcopy(main_net)
            print(total / n_update)
    

from cartpole import CartPoleEnv

if __name__ == "__main__":
    env = CartPoleEnv()
    main_net = DQN()
    target_net = copy.deepcopy(main_net)
    learn_ma_boy(env, main_net, target_net)
    



from torch.autograd import Variable
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
from config import config, user_config

from cartpole import CartPoleEnv

import pickle
from datetime import datetime

import json
from dataloader import Memory

from tqdm import tqdm

def create_net_list(layers, dropout):
    netlist = []
    n = 0
    for i, o in zip(net_config.layers[:-1], net_config.layers[1:]):
        # print("Layer ", n, "[", i, ":", o, "]")
        if (n != 0):
            netlist.append(nn.LeakyReLU())
        netlist.append(nn.Linear(i, o))
        if dropout != 0 and n != 0 and n != len(net_config.layers[:-1]) - 1:
            # print("Dropout")
            netlist.append(nn.Dropout(dropout))
        n += 1
    return netlist


class DQN():
    def __init__(self, layers = net_config.layers, learning_rate=net_config.learning_rate, dropout=net_config.dropout):
        self.model = nn.Sequential(* create_net_list(layers, dropout))
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.model.train()

    def update(self, state, y):
        y_pred = self.model(torch.Tensor(state))
        loss = 0
        for i in range(len(y)):
            loss += self.criterion(y_pred[i], y[i])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        '''
        Why torch no grad ?
        '''
        with torch.no_grad():
            return self.model(torch.Tensor(state))

def choose_action_net(y_pred, epsilon = net_config.epsilon):
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

def reward_optimisation(state, end, reward, modif=True):
    # reward = net_config.reward_loose if end else net_config.reward_win
    reward = net_config.reward_loose if end else reward
    if reward == net_config.reward_win and modif:
        # Angle reward modification
        angle_r = 0.418 / 2
        reward += (((abs(angle_r - abs(state[2]))
                        / angle_r) * 2) - 1) * 2
        # Position reward modification
        pos_r = 0.418 / 2
        reward += (((abs(pos_r - abs(state[0]))
                        / pos_r) * 2) - 1) * 2
    return reward


from plot_data import PlotData
import inspect

class CuteLearning():
    def __init__(self, epsilon=net_config.epsilon, eps_decay=net_config.eps_decay,
                 n_update=net_config.n_update, max_turns=net_config.max_turns,
                 batch=net_config.batch, gamma=net_config.gamma,
                 soft_update=net_config.soft_update, tau=net_config.tau,
                 replay_nb_batch=net_config.replay_nb_batch, dropout=net_config.dropout,
                 reward_optimisation=net_config.reward_optimisation,
                 learning_rate=net_config.learning_rate, layers=net_config.layers,
                 verbose=2, early_stopping=net_config.early_stopping):
        # Reproductibility:
        torch.manual_seed(net_config.seed)
        np.random.seed(net_config.seed)

        # model Hyper Params:
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.n_update = n_update
        self.max_turns = max_turns
        self.batch = batch
        self.gamma = gamma
        self.tau = tau
        self.replay_nb_batch = replay_nb_batch
        self.soft_update = soft_update
        self.reward_optimisation = reward_optimisation
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping

        # Env + Pytorch
        self.cart = CartPoleEnv()
        self.cart.reset()
        self.predi_net = DQN(layers, learning_rate, dropout)
        self.predi_net.model.train()
        self.updat_net = deepcopy(self.predi_net)

        # Data
        self.turn = 0
        self.epidode = 0
        self.consecutive_wins = 0
        self.best_consecutive_wins = 0
        self.last_save = 0
        self.memory = []

        # Visu
        self.plot_data = PlotData()
        self.visu = False
        self.visu_update = user_config.visu_update
        self.visu_window = user_config.visu_window
        self.verbose = verbose

        # print("Initializing classifier:\n")

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        self.my_params = {}
        for arg, val in values.items():
            self.my_params[str(arg)] = val

    
    def learn(self):
        self.predi_net.model.train()
        self.episode = 0
        n = 0
        states = []
        targets = []
        train_iter = list(range(net_config.max_episodes))
        with tqdm(train_iter, desc="CartPole <3 ", unit="episode") as tepisodes:
            _iter = tepisodes if self.verbose == 2 else train_iter
            for _ in _iter:
                self.turn = 0
                end = False
                while not end:
                    # 1. Init
                    state = self.cart.state
                    # 2. Choose action
                    q_values = self.predi_net.predict(state)
                    a = choose_action_net(q_values, self.epsilon)
                    # 3. Perform action
                    next_state, reward, end, _ = self.cart.step(a)
                    # 4. Measure reward
                    reward = reward_optimisation(
                        next_state, end, reward, self.reward_optimisation)
                    q_values_next = self.predi_net.predict(next_state)
                    # 5. Calcul Q-Values
                    q_values[a] = reward + self.gamma * \
                        torch.max(q_values_next).item()

                    self.turn += 1
                    # loader = DataLoader(mem, 2, True)
                    self.memory.append((state, a, next_state, reward, end))
                    # self.updat_net.update(state, q_values)
                    states.append(state)
                    targets.append(q_values)
                    if len(states) >= self.batch:
                        self.updat_net.update(states, targets)
                        states = []
                        targets = []

                    if self.visu:
                        self.cart.render()
                    if self.turn >= self.max_turns:
                        break

                self.episode += 1
                self.replay(self.replay_nb_batch)
                if self.episode % self.n_update == 0 and self.episode:
                    if self.verbose == 1:
                        print("Update")
                    if self.soft_update:
                        self.soft_net_update(self.tau)
                    else:
                        self.predi_net.model.load_state_dict(self.updat_net.model.state_dict())
                # tmp = self.turn
                self.end()
                if user_config.verbose == 2:
                    tepisodes.set_postfix(win=self.consecutive_wins,
                                        w_b=self.best_consecutive_wins,
                                        z_t=int(
                                            self.plot_data.running_mean_big[-1]),
                                        eps=int(self.epsilon * 100))
                if self.consecutive_wins >= net_config.consecutive_wins_required:
                    if self.last_save < self.consecutive_wins:
                        self.save()
                        self.last_save = self.consecutive_wins
                    print(("\nWIN IN " + str(self.episode) + " EPISODES\n"))
                    if net_config.early_stopping:
                        break
                n += 1
        self.cart.close()
        self.plot_data.clear()

    def eval(self, nb_episodes=net_config.eval_episodes):
        self.predi_net.model.eval()
        self.cart.reset()
        self.episode = 0
        train_iter = list(range(nb_episodes))
        vic_list = []
        turns_list = []
        best_win_combo = 0
        win_combo = 0

        with tqdm(train_iter, desc="CartPole eval <3 ", unit="episode") as tepisodes:
            _iter = tepisodes if self.verbose == 2 else train_iter
            for _ in _iter:
                self.turn = 0
                end = False
                while not end:
                    # Choose action
                    q_values = self.predi_net.predict(self.cart.state)
                    a = choose_action_net(q_values, 0)
                    # Perform action
                    _, _, end, _ = self.cart.step(a)
                    self.turn += 1
                    if self.visu:
                        self.cart.render()
                    if self.turn >= self.max_turns:
                        break

                self.cart.reset()
                self.episode += 1

                if self.turn >= net_config.turn_threshold_to_win:
                    vic = 1
                    win_combo += 1
                    if win_combo > best_win_combo:
                        best_win_combo = win_combo
                else:
                    vic = 0
                    win_combo = 0
                vic_list.append(vic)
                turns_list.append(min(self.turn, 200))

                turn_mean = np.mean(turns_list) / 200
                vic_mean = np.mean(vic_list)
                tepisodes.set_postfix(win=best_win_combo,
                                      trn=turn_mean,
                                      vic=vic_mean)

        # print(turns_list, vic_list)
        # for t, v in zip(turns_list, vic_list):
        #     print(t, "\t", v)
        self.turn_mean = np.mean(turns_list) / 200
        self.vic_mean = np.mean(vic_list)
        self.combo = best_win_combo / 200
        # print("turn_mean: ", self.turn_mean)
        # print("vic_mean: ", self.vic_mean)
        # print("combo: ", self.combo)

        val = (self.combo * 5) + (self.vic_mean * 3) + self.turn_mean

        return val 

    def replay(self, batch_nb):
        if (batch_nb * self.batch) > len(self.memory):
            batch_nb = int(len(self.memory) / self.batch)
        data = random.sample(self.memory, batch_nb * self.batch)
        # loader = DataLoader(self.memory, net_config.batch, True)

        states = []
        targets = []
        i = 0
        for state, action, next_state, reward, done in data:
            q_values = self.predi_net.predict(state)
            if done:
                q_values[action] = reward
            else:
                # The only difference between the simple replay is in this line
                # It ensures that next q values are predicted with the target network.
                q_values_next = self.predi_net.predict(next_state)
                q_values[action] = reward + self.gamma * torch.max(q_values_next).item()
            states.append(state)
            targets.append(q_values)
            if len(states) >= self.batch:
                self.updat_net.update(states, targets)
                states = []
                targets = []
                i += 1
            if i == batch_nb:
                break

    def soft_net_update(self, TAU=net_config.tau):
        ''' Update the targer gradually. '''
        # Extract parameters  
        perdi_params = self.predi_net.model.named_parameters()
        updat_params = self.updat_net.model.named_parameters()
        
        updated_params = dict(updat_params)

        for predi_name, predi_param in perdi_params:
            if predi_name in updat_params:
                # Update parameter
                updated_params[predi_name].data.copy_(
                    (TAU)*predi_param.data + (1-TAU)*updat_params[predi_param].data)

        self.predi_net.model.load_state_dict(updated_params)
        self.updat_net.model.load_state_dict(self.predi_net.model.state_dict())
        # self.updat_net.update(state, q_values)


    def end(self):
        self.plot_data.new_data(self.turn)
        if self.turn > net_config.turn_threshold_to_win:
            self.consecutive_wins += 1
            if self.best_consecutive_wins < self.consecutive_wins:
                self.best_consecutive_wins = self.consecutive_wins
        else:
            self.consecutive_wins = 0
            if self.last_save * 1.2 < self.best_consecutive_wins and 50 <= self.best_consecutive_wins:
                self.save()
                self.last_save = self.best_consecutive_wins
        if self.verbose == 1:
            print("Episode: ", self.episode, "\tTurn:",
                self.turn, "\tEpsilon:", self.epsilon,
                "\tWins: ", "{:3}".format(self.consecutive_wins),
                "/", self.best_consecutive_wins)
        self.turn = 0
        self.cart.reset()
        if user_config.plot:
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
        if self.verbose == 1:
            print("Saving")
        name = "model_cache/"
        name += str(self.best_consecutive_wins) + "Wins"
        name += "_"
        name += str(self.episode) + "Episodes"
        name += "_"
        now = datetime.now()
        name += now.strftime("%d-%m %H:%M")
        with open(name + ".json", "w+") as f:
            json.dump([self.my_params, config, net_config], f, indent=4,
                      default=lambda o: '<not serializable>')
        with open(name + ".mdl", "wb+") as f:
            pickle.dump(self, f)
        self.plot_data.save(name)

if __name__ == "__main__":
    Cutie = CuteLearning()
    Cutie.learn()
    res = Cutie.eval()
    print(res)

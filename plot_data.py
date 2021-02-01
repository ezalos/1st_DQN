import matplotlib.pyplot as plt
import numpy as np
from config import config

class PlotData():
    def __init__(self, running_mean_len = config.graph_running_mean):
        plt.ion()
        self.points = []
        self.running_mean = []
        self.running_mean_len = running_mean_len

    def new_data(self, new_point):
        self.points.append(new_point)
        self.running_mean.append(np.mean(self.points[-self.running_mean_len:]))
        pass

    def clear(self):
        plt.ioff()
        self.points = []
        self.running_mean = []

    def graph(self):
        if False:
            plt.scatter(list(range(len(data))), data, s=1, c="b")
        plt.plot(self.running_mean, c="r")
        # plt.autoscale_view()
        plt.pause(0.05)
        plt.show()

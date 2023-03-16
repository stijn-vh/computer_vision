import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


from scipy.signal import savgol_filter

class TrajectoryPlotter():
    X = [[], [], [], []]
    Y = [[], [], [], []]

    graphs = []

    firstDraw = True

    def __init__(self, params) -> None:
        self.xb = params['xb']
        self.zb = params['zb']

        
    def initialize_graphs(self, new_centers):
        for i in range(len(new_centers)):
            self.X[i].append(new_centers[i][0])
            self.Y[i].append(new_centers[i][1])

            graph = plt.plot(self.X[i], self.Y[i], label = 'Person ' + str(i))[0]
            self.graphs.append(graph)

        plt.legend()

        plt.xlim([-self.xb,self.xb])
        plt.ylim([-self.zb,self.zb])
        plt.ion()

        self.firstDraw = False

    def append_new_points_to_plot(self, new_centers):
        for i in range(len(new_centers)):
            self.X[i].append(new_centers[i][0])
            self.Y[i].append(new_centers[i][1])

            smooth_X, smooth_Y = self.get_smooth_trajectories(self.X[i], self.Y[i])

            self.graphs[i].set_xdata(smooth_X)
            self.graphs[i].set_ydata(smooth_Y)

    def add_to_plot(self, new_centers):
        if (self.firstDraw == True):
            self.initialize_graphs(new_centers)
        else:
            self.append_new_points_to_plot(new_centers)

        plt.draw()
        plt.pause(0.01)

    def get_smooth_trajectories(self, X, Y):
        div_window_length = 3
        polyorder = 6
        
        if (int(len(X) / div_window_length) <= polyorder):
            return X, Y
        
        X = savgol_filter(X, int(len(X) / div_window_length), polyorder).tolist()
        Y = savgol_filter(Y, int(len(Y) / div_window_length), polyorder).tolist()

        return X, Y



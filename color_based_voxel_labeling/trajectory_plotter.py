import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    
            graph = plt.scatter(self.X[i], self.Y[i], label = 'Person ' + str(i))[0]
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

            self.graphs[i].set_xdata(self.X[i])
            self.graphs[i].set_ydata(self.Y[i])

    def add_to_plot(self, new_centers):
        if (self.firstDraw == True):
            self.initialize_graphs(new_centers)
        else:
            self.append_new_points_to_plot(new_centers)

        plt.draw()
        plt.pause(0.001)



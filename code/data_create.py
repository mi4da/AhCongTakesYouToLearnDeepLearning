import matplotlib.pyplot as plt
import numpy as np


class DataCreater:
    def __init__(self):
        self.x = np.arange(0, 100)
        self.y = np.sin(1 / 10 * self.x) + 1

    def normliza_x(self):
        self.x = self.x / 100

    def get_data(self):
        return (self.x, self.y)

    def plot_data(self):
        plt.scatter(self.x,self.y,color = 'b')
        plt.plot(self.x, self.y, color='b')


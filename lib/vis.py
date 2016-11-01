import matplotlib.pyplot as plt
import numpy as np
import time
class Visualiser(object):

    def __init__(self):
        self.fig = plt.figure()
        self.x_label="x-Axis"
        self.y_label="y-Axis"
        self.title="Title"

    def switch_to_figure(self):
        plt.figure(self.fig.number)

    def show_labels(self):
        self.switch_to_figure()
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)

    def update_fig(self):
        self.switch_to_figure()

    def add_data(self, data):
        self.data=data

    def show(self):
        self.switch_to_figure()
        plt.clf()
        self.show_labels()
        self.update_fig()
        self.fig.canvas.draw()
        plt.show(block=False)

    def save(self, path):
        self.switch_to_figure()
        plt.savefig(path)


class ImageVisualiser(Visualiser):
    def __init__(self, title, size, cmap="Greys"):
        Visualiser.__init__(self)
        self.x_label = ""
        self.y_label = ""
        self.title = title
        self.data = np.zeros(size)
        self.cmap = cmap

    def update_fig(self):
        self.switch_to_figure()
        plt.imshow(self.data, cmap=self.cmap)

class ErrorVisualiser(Visualiser):
    def __init__(self, title, num_lines,line_names=None, y_log=True, y_lim=(10**-5,10**5)):
        Visualiser.__init__(self)
        self.x_label = "Epoch"
        self.y_label = "Error"
        self.y_log = y_log
        self.title = title
        self.num_lines = num_lines
        self.line_names = line_names
        self.data = [[1] for i in range(num_lines)]
        self.plots=None
        self.ylim = y_lim

    def add_data(self, data):
        for i in range(self.num_lines):
            self.data[i].append(data[i])

    def show_labels(self):
        self.switch_to_figure()
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        if self.line_names!=None and self.plots!=None:
            plt.legend(self.plots, self.line_names)

    def update_fig(self):
        self.switch_to_figure()
        if self.y_log:
            plt.yscale('log')
            plt.ylim(self.ylim)
        pData = []
        for i in self.data:
            pData.append(range(len(i)))
            pData.append(i)
        self.plots = plt.plot(*pData)

class ScatterVisualiser(Visualiser):

    def __init__(self):
        Visualiser.__init__(self)

    def add_data(self, data):
        self.data = data

    def update_fig(self):
        self.switch_to_figure()
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.scatter(self.data[:,0],self.data[:,1])

"""
Example of Use:
"""

"""
v = ImageVisualiser("Test",(10,10))
e = ErrorVisualiser("ETest",2,["updownup","uuuup"])
for i in range(10):
    for j in range(10):
        d = np.zeros((10,10))
        d[i][j]=1
        v.add_data(d)
        v.show()
        e.add_data([j,i])
        e.show()
        time.sleep(0.1)
"""
import numpy as np
import matplotlib as mpl
from scipy import linalg
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class ClassificationPlot:
    def __init__(self, title_=None, x1lab_=None, x2lab_=None, size_=(1,1)):
        # === canvas ===
        self.fig = plt.figure()                      # the reference of our figure
        self.splot = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

        # === labels ===
        self.title = title_
        self.x1lab = x1lab_
        self.x2lab = x2lab_

    def label_title(self, title_):
        self.title = title_

    def label_axis(self, x1lab_, x2lab_):
        self.x1lab = x1lab_
        self.x2lab = x2lab_

    def plot_data_p2(self, Xp2_, y_, classifier_, grid_step_=0.01):
        x1 = Xp2_[:, 0];  std1 = np.std(x1)
        x2 = Xp2_[:, 1];  std2 = np.std(x2)
        X1, X2 = np.meshgrid(np.arange(start=x1.min() - std1, stop=x1.max() + std1, step=grid_step_),
                             np.arange(start=x2.min() - std2, stop=x2.max() + std2, step=grid_step_))
        x12_grid = np.array([X1.ravel(), X2.ravel()]).T
        plt.contourf(X1, X2, classifier_.predict(x12_grid).reshape(X1.shape),
                     alpha=0.2, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_)):
            plt.scatter(Xp2_[y_ == j, 0], Xp2_[y_ == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title(self.title)
        plt.xlabel(self.x1lab)
        plt.ylabel(self.x2lab)
        plt.legend()

    def plot_ellipse(self, mean, cov, color):
        """
        :param mean: numpy array, such as [0 , 0]
        :param cov:  numpy matrix, such as [[1,0] [0,1]]
        :param color: the color filled in the ellipse
        """
        v, w = linalg.eigh(cov)
        u = w[0] / linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        # filled Gaussian at 2 standard deviation
        ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                                  180 + angle, facecolor=color,
                                  edgecolor='yellow',
                                  linewidth=2, zorder=2)
        ell.set_clip_box(self.splot.bbox)
        ell.set_alpha(0.3)
        self.splot.add_artist(ell)
        plt.legend()


def main():
    Xp2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
    x1 = Xp2[:, 0];
    std1 = np.std(x1)
    x2 = Xp2[:, 1];
    std2 = np.std(x2)
    X1, X2 = np.meshgrid(np.arange(start=x1.min() - std1, stop=x1.max() + std1, step=0.01),
                         np.arange(start=x2.min() - std2, stop=x2.max() + std2, step=0.01))
    x12_grid = np.array([X1.ravel(), X2.ravel()]).T


    print("end")



if __name__ == "__main__":
    main()
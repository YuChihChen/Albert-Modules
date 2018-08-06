import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, gd_method_='batch', lr_method_=None, 
                 eta_ = 1, n_iterations_ = 1000, theta0_=None, minibatch_size_=25, 
                 cost_fun_ = None, steps_cost_ = 1, 
                 gradient_fun_=None, *args, **kwargs):
        self.gd_method = gd_method_           # gradient descent method
        self.lr_method = lr_method_           # learning rate method
        self.eta = eta_
        self.n_iterations = n_iterations_
        self.theta = theta0_
        self.minibatch_size = minibatch_size_
        self.gd_fun = gradient_fun_
        self.ct_fun = cost_fun_
        self.steps_cost = steps_cost_
        self.lr_fun = None
        self.args = args
        self.kwargs = kwargs
        self.__i = None
        self.__gradients_sum = None
        self.__thetas = None
        self.__costs = None
        self.opt_theta = None
    
    # ========== I. Common Functions ==========
    @staticmethod
    def __graident_mse(theta_, X_, y_):
        size = X_.shape[0]
        return (2 / size) * X_.T.dot(X_.dot(theta_) - y_)
    
    def __cost_mse(theta_, X_, y_):
        resi = X_.dot(theta_) - y_ 
        return np.mean(resi * resi)
    
    def __set_theta0(self, X_):
        size = X_.shape[1]
        if self.theta is None:
            self.theta = np.array([0]*size)
        else:
            self.theta = np.array(self.theta)
        self.theta = self.theta.reshape((size, 1))
    
    def __set_gd_function(self):
        if self.gd_fun is None:
            self.gd_fun = self.__graident_mse
            self.args = ()
            self.kwargs = {}
    
    def __set_lr_function(self):
        if self.lr_method is None:
            self.lr_fun = self.lr_unit
        elif self.lr_method == 'adaptive':
            self.lr_fun = self.lr_adptive
        elif self.lr_method == 'adagrad':
            self.lr_fun = self.lr_adagrad
        else:
            raise ValueError('{} lr_method is not avaliable'.format(self.lr_method))
    
    # ========== II. Learning Rate Functions ==========
    def lr_unit(self):
        return 1
    
    def lr_adptive(self):
        return (self.__i + 1)**(-0.5)
    
    def lr_adagrad(self):
        return (self.__gradients_sum)**(-0.5)
    
    # ========== III. Gradient Methods ==========
    def __gd_initilization(self, X_, y_):
        self.__set_theta0(X_)
        self.__set_gd_function()
        self.__set_lr_function()
        self.__gradients_sum = np.ones(shape=(X_.shape[1], 1))
        self.__thetas = list()
        self.__costs = list()
        self.opt_theta = None
    
    def __theta_update(self, X_, y_):
        gradient = self.gd_fun(self.theta, X_, y_, *self.args, **self.kwargs)
        if self.ct_fun is not None and (self.__i % self.steps_cost == 0):
            cost = self.ct_fun(self.theta, X_, y_, *self.args, **self.kwargs)
            self.__costs.append(cost)
        self.__gradients_sum = self.__gradients_sum + (gradient ** 2)
        self.__thetas.append(self.theta)
        self.theta = (self.theta - self.eta * self.lr_fun() * gradient)
    
    def __batch(self, X_, y_):
        self.__gd_initilization(X_, y_)
        for i in range(self.n_iterations):
            self.__i = i
            self.__theta_update(X_, y_)
        self.opt_theta = self.theta
    
    def __stochastic(self, X_, y_):
        self.__gd_initilization(X_, y_)
        size = X_.shape[0]
        for i in range(self.n_iterations):
            for r in range(size):
                random_index = np.random.randint(size)
                xr = X_[random_index:random_index+1, :]
                yr = y_[random_index:random_index+1]
                self.__i = r + i * size
                self.__theta_update(xr, yr)
        self.opt_theta = self.theta
    
    def __minibatch(self, X_, y_):
        self.__gd_initilization(X_, y_)
        size = X_.shape[0]
        for i in range(self.n_iterations):
            shuffled_indices = np.random.permutation(size)
            X_shuffled = X_[shuffled_indices]
            y_shuffled = y_[shuffled_indices]
            for r in range(0, size, self.minibatch_size): 
                xr = X_shuffled[r:r + self.minibatch_size]
                yr = y_shuffled[r:r + self.minibatch_size]
                self.__i = (r + i * size) // self.minibatch_size
                self.__theta_update(xr, yr)
        self.opt_theta = self.theta
        
    
    # ========== IV. Plots ==========
    def plot_path(self, idx_x_=0, idx_y_=1):
        thetas = np.array(self.__thetas)
        plt.figure(figsize=(10,8))
        plt.plot(thetas[:, idx_x_], thetas[:, idx_y_], "g-s", linewidth=1)
        plt.scatter(self.opt_theta[idx_x_,:], self.opt_theta[idx_y_,:],
                    color='r', marker='*', s=300, zorder=10)
        plt.xlabel(r"$\theta_0$", fontsize=20)
        plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
        plt.title('{} gradient descent'.format(self.gd_method), fontsize=20)
    
    def plot_cost(self):
        if self.ct_fun is None:
            raise ValueError('Error! Cost function is None.')
        iters = range(len(self.__costs))
        plt.figure(figsize=(10,8))
        plt.scatter(iters, self.__costs)
        plt.xlabel("iterations (per {} steps)".format(self.steps_cost), fontsize=20)
        plt.ylabel("cost", fontsize=20)
        plt.title('gd_method = {}, lr_method = {}, eta = {}'
                  .format(self.gd_method, self.lr_method, self.eta), fontsize=20)
        
    
    # ========== V. The Fitting Function ==========
    def fit(self, X_, y_):
        if self.gd_method == 'batch':
            self.__batch(X_, y_)
        elif self.gd_method == 'stochastic':
            self.__stochastic(X_, y_)
        elif self.gd_method == 'minibatch':
            self.__minibatch(X_, y_)
        else:
            raise ValueError('{} gd_method is not avaliable'.format(self.gd_method))
        return self.opt_theta

def cost_mse(theta_, X_, y_):
    resi = X_.dot(theta_) - y_ 
    return np.mean(resi * resi)

def main():
    print('\n============ I. Generate and Plot Data ============')
    size  = 100
    x = 2 * np.random.rand(size)
    y_true = 4 + 3 * x
    y = y_true + np.random.randn(size)
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y)
    plt.axis([0, 2, 0, 15])
    plt.show()
    df = pd.DataFrame(np.array([np.ones((size)), x, y]).T, columns=['x0', 'x1', 'y'])
    X = df.loc[:, ['x0', 'x1']].values
    y = df.loc[:, ['y']].values
    print('Exact Solution: y = b0 + b1 * x1 + noise, where b0 = 4, b1 = 3')
    
    
    print('\n============ II. Gradient Descient Method ============')
    # --- 1. batch gradient descent ---
    gd = GradientDescent(eta_=2, lr_method_='adagrad', n_iterations_=50,
                         cost_fun_=cost_mse)
    theta = gd.fit(X, y)
    gd.plot_path(); plt.show()
    gd.plot_cost(); plt.show()
    print('batch gradient descent: (b0, b1) = {}'.format(theta.T[0]))
    # --- 2. stochastic gradient descent ---
    gd = GradientDescent(eta_=2, gd_method_='stochastic', lr_method_='adagrad', 
                         n_iterations_=10, cost_fun_=cost_mse)
    theta = gd.fit(X, y)
    gd.plot_path(); plt.show()
    gd.plot_cost(); plt.show()
    print('stochastic gradient descent: (b0, b1) = {}'.format(theta.T[0]))
    # --- 3. minibatch gradient descent ---
    gd = GradientDescent(eta_=2, gd_method_='minibatch', lr_method_='adagrad', 
                         minibatch_size_=20, n_iterations_=10, cost_fun_=cost_mse)
    theta = gd.fit(X, y)
    gd.plot_path(); plt.show()
    gd.plot_cost(); plt.show()
    print('minibatch gradient descent: (b0, b1) = {}'.format(theta.T[0]))
    
if __name__ == '__main__':
    main()
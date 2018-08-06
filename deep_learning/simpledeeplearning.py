import numpy as np
import matplotlib.pyplot as plt


class SimpleDeepLearning:
    def __init__(self, nn_sizes_list_=None, act_hidden_='tanh', act_output_='sigmoid',
                 epochs_ = 20, batch_size_=20, eta_ = 1, steps_cost_ = 1,):
        if nn_sizes_list_ is None:
            raise ValueError('Error! nn_sizes_list_ is None')
        self.nn_sizes_list = nn_sizes_list_
        self.act_hidden = act_hidden_
        self.act_output = act_output_
        self.epochs = epochs_
        self.batch_size = batch_size_
        self.eta = eta_
        self.steps_cost = steps_cost_
        self.L = len(self.nn_sizes_list)
        self.O = None
        self.Z = None
        self.F = None
        self.B = None
        self.actfun_hidden = None
        self.actdev_hidden = None
        self.actfun_output = None
        self.actdev_output = None
        self.costs = None


    # ==================== Z. Activatioin Functions ====================
    # --- 0. test funtion ---
    @staticmethod
    def __act_test(Z_):
        return Z_ * 2
    
    @staticmethod
    def __dev_test(Z_):
        return 2
    
    # --- 1. sigmoid funtion ---
    @staticmethod
    def __sigmoid(Z_):
        return 1 / (1 + np.exp(-Z_))
    
    @staticmethod
    def __sigmoid_devrivate(Z_):
        return np.exp(-Z_) / ((1 + np.exp(-Z_)) ** 2)
    
    # --- 2. tanh funtion ---
    @staticmethod
    def __tanh(Z_):
        return np.tanh(Z_)
    
    @staticmethod
    def __tanh_devrivate(Z_):
        return 1 - np.power(np.tanh(Z_), 2)
    
    # --- 9. cost functions ---
    @staticmethod
    def __cost_mlm(sigma_, y_):
        m = y_.shape[0]
        cost = - (1 / m) * (y_.T.dot(np.log(sigma_)) + (1-y_).T.dot(np.log(1 - sigma_)))
        return np.squeeze(cost)
    
    # ==================== I. Build-up a Neural Netwrok ====================
    def __create_testO(self):
        self.O = list()
        self.O.append(np.array([[1,2,3], [4,5,6], [7,8,9]]))
        self.O.append(np.array([[1], [3], [5]]))
        self.O.append(None)
    
    def __create_omega(self):
        """ does not depend on Xb input """
        self.O = list()
        for l in range(self.L - 1):
            ni = self.nn_sizes_list[l]
            nj = self.nn_sizes_list[l + 1]
            if l < self.L - 2:
                self.O.append(np.zeros((ni + 1, nj + 1)))
                self.O[l][0 ,  :] = 0  # the settings are for checking wiht coursera
                self.O[l][: , 0 ] = 0
                self.O[l][1:, 1:] = (np.random.randn(nj, ni) * 0.01).T
            else:
                self.O.append(np.zeros((ni + 1, nj)))
                self.O[l][0 ,  :] = 0
                self.O[l][1:,  :] = (np.random.randn(nj, ni) * 0.01).T
        self.O.append(None)
        
    
    # ==================== II. Forward/Backward Propogation ====================
    def __build_OZFB(self, data_size_):
        """ depends on data size of Xb input """
        self.Z = list()
        self.F = list()
        self.B = list()
        for l in range(self.L):
            nl = self.nn_sizes_list[l]
            if l < self.L - 1:
                self.Z.append(np.zeros((data_size_, nl + 1)))
                self.F.append(np.zeros((data_size_, nl + 1)))
                self.B.append(np.zeros((data_size_, nl + 1)))
            else:
                self.Z.append(np.zeros((data_size_, nl)))
                self.F.append(np.zeros((data_size_, nl)))
                self.B.append(np.zeros((data_size_, nl)))
          
    def __Forward(self, actfun_hidden_, actfun_output_, Xb_):
        self.F[0] = Xb_
        for l in range(1, self.L - 1):
            self.Z[l] = self.F[l-1].dot(self.O[l-1])
            self.F[l] = actfun_hidden_(self.Z[l])
            self.F[l][:, 0] = 1
        self.Z[-1] = self.F[-2].dot(self.O[-2])
        self.F[-1] = actfun_output_(self.Z[-1])

    def __Backward(self, devfun_hidden_, devfun_output_, y_):
        sa = self.F[-1]
        self.B[-1] = devfun_output_(self.Z[-1]) * ( sa - y_) / (sa * (1 - sa))
        for l in range(self.L-2, 0, -1):
            self.B[l] = devfun_hidden_(self.Z[l]) * (self.B[l+1].dot(self.O[l].T))
     
    def __Forward_Backward(self, X_, y_):
        data_size = X_.shape[0] 
        Xb = np.c_[np.ones((data_size, 1)), X_]
        self.__build_OZFB(data_size)
        self.__Forward(self.actfun_hidden, self.actfun_output, Xb)
        self.__Backward(self.actdev_hidden, self.actdev_output, y_)
        
        
    # ==================== III. Gradient Descent ====================
    def __omegas_update(self, data_size_):
        for l in range(self.L - 1):
            Fa = self.F[l]
            Bb = self.B[l + 1]
            gradient_l = (1 / data_size_) * Fa.T.dot(Bb)
            self.O[l] = (self.O[l] - self.eta * gradient_l)
            if l < self.L - 2:
                self.O[l][: , 0 ] = 0
            
    def __minibatch(self, X_, y_):
        data_size = X_.shape[0]
        for i in range(self.epochs):
            shuffled_indices = np.random.permutation(data_size)
            X_shuffled = X_[shuffled_indices]
            y_shuffled = y_[shuffled_indices]
            for r in range(0, data_size, self.batch_size): 
                xr = X_shuffled[r:r + self.batch_size]
                yr = y_shuffled[r:r + self.batch_size]
                self.__Forward_Backward(xr, yr)
                self.__omegas_update(data_size)
                iterations = (r + i * data_size) // self.batch_size
                if iterations % self.steps_cost == 0:
                    cost = self.__cost_mlm(self.F[-1], yr)
                    self.costs.append(cost)
    
    # ==================== IV. Initialization Before Fitting ====================   
    def __init_funs_hidden(self):
        if self.act_hidden == 'sigmoid':
            self.actfun_hidden = self.__sigmoid
            self.actdev_hidden = self.__sigmoid_devrivate
        elif self.act_hidden == 'tanh':
            self.actfun_hidden = self.__tanh
            self.actdev_hidden = self.__tanh_devrivate
        elif self.act_hidden == 'test':
            self.actfun_hidden = self.__act_test
            self.actdev_hidden = self.__dev_test
        else:
            raise ValueError('act_hidden = {} is not available'.format(self.act_hidden))
    
    def __init_funs_output(self):
        if self.act_output == 'sigmoid':
            self.actfun_output = self.__sigmoid
            self.actdev_output = self.__sigmoid_devrivate
        elif self.act_output == 'test':
            self.actfun_output = self.__act_test
            self.actdev_output = self.__dev_test
        else:
            raise ValueError('act_hidden = {} is not available'.format(self.act_hidden))          
        
    def __initializaiton(self):
        np.random.seed(2)
        self.__init_funs_hidden()
        self.__init_funs_output()
        self.__create_omega()
        self.costs = list()
    
    
    # ==================== V. Fitting Function ====================
    def fit(self, X_, y_):
        self.__initializaiton()
        self.__minibatch(X_, y_)
        
    def predict(self, X_):
        data_size = X_.shape[0] 
        Xb = np.c_[np.ones((data_size, 1)), X_]
        self.__build_OZFB(data_size)
        self.__Forward(self.actfun_hidden, self.actfun_output, Xb)
        return (self.F[-1] > 0.5) * 1
     
        
    # ==================== VI. Plot Functions ====================
    def plot_cost(self):
        iters = range(len(self.costs))
        plt.figure(figsize=(10,8))
        plt.scatter(iters, self.costs)
        plt.xlabel("iterations (per {} steps)".format(self.steps_cost), fontsize=20)
        plt.ylabel("cost", fontsize=20)
        plt.title('learning rate = {}'.format(self.eta), fontsize=20)
    
    
        
def main():
    sdl = SimpleDeepLearning(nn_sizes_list_=[2, 4, 1])
    np.random.seed(1)
    X = np.random.randn(2, 3).T
    y = np.array([[1.74481176], [-0.7612069], [0.3190391]])
    sdl.fit(X, y)
    sdl.plot_cost()
    plt.show()
    y_pred = sdl.predict(X)
    print(y_pred)
    
    
    
if __name__ == '__main__':
    main()
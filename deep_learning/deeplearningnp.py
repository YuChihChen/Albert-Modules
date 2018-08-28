import time
import numpy as np
import matplotlib.pyplot as plt


class DeepLearningNP:
    def __init__(self, nn_sizes_list_=None, act_hidden_='tanh', act_output_='sigmoid', cost_='mlm', 
                 epochs_=2000, batch_size_=20, eta_=0.0075, steps_cost_=100,
                 lambd_=0, dropout_probs_=None, adam_=False, gd_check_=False):
        if nn_sizes_list_ is None:
            raise ValueError('Error! nn_sizes_list_ is None')
        # --- input attributes ---
        self.nn_sizes_list = nn_sizes_list_
        self.act_hidden = act_hidden_
        self.act_output = act_output_
        self.cost = cost_
        self.epochs = epochs_
        self.batch_size = batch_size_
        self.eta = eta_
        self.steps_cost = steps_cost_
        self.lambd =lambd_
        self.dropout_probs = dropout_probs_
        self.adam = adam_
        self.gd_check = gd_check_
        # -- class attributes ---
        self.L = len(self.nn_sizes_list)
        self.O = None   # omega and b
        self.Z = None   # X dot O + b
        self.F = None   # sigma(Z)
        self.B = None   # partial_C / partial_Z
        self.D = None   # Dropdout matrix
        self.G = None   # gradients
        self.V = None   # for momentum method 
        self.S = None   # for RMS method
        self.Vbeta = 0
        self.Sbeta = 0
        self.i = 0
        self.actfun_hidden = None
        self.actdev_hidden = None
        self.actfun_output = None
        self.actdev_output = None
        self.costfun = None
        self.costdev = None
        self.costs = None
        self.yr = None
    
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
                self.O[l][0 ,  :] = 0   # the settings are for checking with coursera
                self.O[l][: , 0 ] = 0
                self.O[l][1:, 1:] = (np.random.randn(nj, ni) * np.sqrt(1 / ni)).T
            else:
                self.O.append(np.zeros((ni + 1, nj)))
                self.O[l][0 ,  :] = 0
                self.O[l][1:,  :] = (np.random.randn(nj, ni) * np.sqrt(1 / ni)).T
        self.O.append(None)             # makes O having same length as Z, F, B
        self.V = list()
        self.S = list()
        for l in range(self.L - 1):
            self.V.append(self.O[l].copy())
            self.S.append(self.O[l].copy())
            self.V[l] = 0
            self.S[l] = 0
        self.V.append(None)
        self.S.append(None)
        
    
    # ==================== II. Forward/Backward Propogation ====================
    def __build_ZFB(self, data_size_):
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
      
    def __create_dropout_d(self, dropout_=True):
        np.random.seed(1)
        self.D = list()
        if self.dropout_probs is not None:
            assert(self.dropout_probs[ 0] == 1)
            assert(self.dropout_probs[-1] == 1)
        for l in range(self.L):
            nm = self.F[l].shape[0]
            nj = self.F[l].shape[1]
            Dl = np.ones((nm, nj))
            if self.dropout_probs and dropout_ and (l>0) and (l<self.L-1):
                Dl_sub = np.random.rand(nj - 1, nm).T
                Dl[:, 1:] = Dl_sub
                Dl = (Dl <= self.dropout_probs[l])
                Dl[:, 0] = True
            self.D.append(Dl)
        
    def __Forward(self, actfun_hidden_, actfun_output_, Xb_):
        self.F[0] = Xb_
        for l in range(1, self.L - 1):
            self.Z[l] = self.F[l-1].dot(self.O[l-1])
            self.F[l] = actfun_hidden_(self.Z[l])
            self.F[l][:, 0] = 1
            self.F[l] = self.F[l] * self.D[l]
            if self.dropout_probs:
                self.F[l][:, 1:] /= self.dropout_probs[l]
        self.Z[-1] = self.F[-2].dot(self.O[-2])
        self.F[-1] = actfun_output_(self.Z[-1]) 

    def __Backward(self, devfun_hidden_, devfun_output_, y_):
        sa = self.F[-1]
        self.B[-1] = devfun_output_(self.Z[-1]) * self.costdev(sa, y_)
        for l in range(self.L-2, 0, -1):
            self.B[l] = devfun_hidden_(self.Z[l]) * (self.B[l+1].dot(self.O[l].T))
            self.B[l] = self.B[l] * self.D[l] 
            if self.dropout_probs:
                self.B[l][:, 1:] /= self.dropout_probs[l]
        for l in range(self.L):
            self.B[l][np.isnan(self.B[l])] = 0          # force zero
     
    def __Forward_Backward(self, X_, y_):
        data_size = X_.shape[0] 
        Xb = np.c_[np.ones((data_size, 1)), X_]
        self.__build_ZFB(data_size)
        self.__create_dropout_d()
        self.__Forward(self.actfun_hidden, self.actfun_output, Xb)
        self.__Backward(self.actdev_hidden, self.actdev_output, y_)
                                              
    
    # ==================== III. Gradient Descent ====================
    def __omegas_update(self):
        data_size = self.Z[0].shape[0]
        gradients_reg = self.__L2_derivative(data_size)
        self.G = list()
        for l in range(self.L - 1):
            Fa = self.F[l]
            Bb = self.B[l + 1]
            gradient_l = Fa.T.dot(Bb)
            gradient_l_reg = gradients_reg[l]
            gradient_l[np.isnan(gradient_l)] = 0               # force zero
            gradient_l += gradient_l_reg
            self.G.append(gradient_l)
            self.V[l] = self.Vbeta * self.V[l] + (1 - self.Vbeta) * gradient_l
            self.S[l] = self.Sbeta * self.S[l] + (1 - self.Sbeta) * (gradient_l * gradient_l)
        self.G.append(None)
        if self.gd_check:
            norm = self.__GD_check()
            if norm > 1e-5:
                print('In gradient check, norm = {}'.format(norm))
        for l in range(self.L - 1):
            Vcor = self.V[l] / (1 - (self.Vbeta ** (self.i+1)))
            Scor = self.S[l] / (1 - (self.Sbeta ** (self.i+1)))
            if not self.adam:
                Scor = 1
            self.O[l] = (self.O[l] - self.eta * (1 / np.sqrt(Scor + 1e-8)) * Vcor)
            if l < self.L - 2:
                self.O[l][: , 0 ] = 0
            
    def __minibatch(self, X_, y_):
        np.random.seed(1)
        data_size = X_.shape[0]
        m = min(data_size, self.batch_size)
        for i in range(self.epochs + 1):
            shuffled_indices = np.random.permutation(data_size)
            X_shuffled = X_[shuffled_indices]
            y_shuffled = y_[shuffled_indices]
            for r in range(0, data_size, self.batch_size):
                self.i += 1
                xr = X_shuffled[r:r + self.batch_size]
                yr = y_shuffled[r:r + self.batch_size]
                self.yr = yr
                self.__Forward_Backward(xr, yr)            
                iterations = (r + i * data_size) // m
                if iterations % self.steps_cost == 0:
                    cost = self.costfun(self.F[-1], yr) + self.__L2_cost(xr.shape[0])
                    self.costs.append(cost)
                    print('epoch={}, cost={}'.format(i, cost))            
                self.__omegas_update()
                
    # ==================== IV. Initialization Before Fitting ====================   
    def __init_funs_hidden(self):
        if self.act_hidden == 'sigmoid':
            self.actfun_hidden = self.__sigmoid
            self.actdev_hidden = self.__sigmoid_derivative
        elif self.act_hidden == 'tanh':
            self.actfun_hidden = self.__tanh
            self.actdev_hidden = self.__tanh_derivative
        elif self.act_hidden == 'relu':
            self.actfun_hidden = self.__relu
            self.actdev_hidden = self.__relu_derivative
        elif self.act_hidden == 'test':
            self.actfun_hidden = self.__act_test
            self.actdev_hidden = self.__dev_test
        else:
            raise ValueError('act_hidden = {} is not available'.format(self.act_hidden))
    
    def __init_funs_output(self):
        if self.act_output == 'sigmoid':
            self.actfun_output = self.__sigmoid
            self.actdev_output = self.__sigmoid_derivative
        elif self.act_output == 'test':
            self.actfun_output = self.__act_test
            self.actdev_output = self.__dev_test
        else:
            raise ValueError('act_hidden = {} is not available'.format(self.act_hidden))

    def __init_funs_cost(self):
        if self.cost == 'mlm':
            self.costfun = self.__cost_mlm
            self.costdev = self.__cost_derivative_mlm
        elif self.cost == 'corr':
            self.costfun = self.__cost_corr
            self.costdev = self.__cost_derivative_corr
        else:
            raise ValueError('cost_ = {} is not available'.format(self.cost))
        
    def __initializaiton(self):
        np.random.seed(3)
        self.__init_funs_hidden()
        self.__init_funs_output()
        self.__init_funs_cost()
        self.__create_omega()
        self.costs = list()
        if self.adam:
            self.Vbeta = 0.9
            self.Sbeta = 0.999
    
    
    # ==================== V. Fitting Function ====================
    def fit(self, X_, y_):
        self.__initializaiton()
        self.__minibatch(X_, y_)
        
    def predict(self, X_):
        data_size = X_.shape[0] 
        Xb = np.c_[np.ones((data_size, 1)), X_]
        self.__build_ZFB(data_size)
        self.__create_dropout_d(dropout_=False)
        self.__Forward(self.actfun_hidden, self.actfun_output, Xb)
        return (self.F[-1] > 0.5) * 1

    def scores(self, X_):
        data_size = X_.shape[0]
        Xb = np.c_[np.ones((data_size, 1)), X_]
        self.__build_ZFB(data_size)
        self.__create_dropout_d(dropout_=False)
        self.__Forward(self.actfun_hidden, self.actfun_output, Xb)
        return self.F[-1]
     
        
    # ==================== VI. Plot Functions ====================
    def plot_cost(self):
        iters = range(len(self.costs))
        plt.figure(figsize=(10,8))
        plt.plot(iters, self.costs)
        plt.xlabel("iterations (per {} steps)".format(self.steps_cost), fontsize=20)
        plt.ylabel("cost", fontsize=20)
        plt.title('learning rate = {}'.format(self.eta), fontsize=20)
    
    
    
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
    def __sigmoid_derivative(Z_):
        return np.exp(-Z_) / ((1 + np.exp(-Z_)) ** 2)
    
    # --- 2. tanh funtion ---
    @staticmethod
    def __tanh(Z_):
        return np.tanh(Z_)
    
    @staticmethod
    def __tanh_derivative(Z_):
        return 1 - np.power(np.tanh(Z_), 2)
    
    # --- 3. relu funtion ---
    @staticmethod
    def __relu(Z_):
        Z_[np.isnan(Z_)] = 0
        return np.maximum(0, Z_)
    
    @staticmethod
    def __relu_derivative(Z_):
        A = np.ones(Z_.shape)
        A[Z_ <= 0] = 0
        return A
    
    # --- 9. cost functions ---
    @staticmethod
    def __cost_mlm(sigma_, y_):
        m = y_.shape[0]
        sigma_[(sigma_ == 1) | (sigma_ == 0)] = np.nan
        ls1 = np.log(sigma_); ls1[np.isnan(ls1) | np.isinf(ls1)] = 0
        ls2 = np.log(1 - sigma_); ls2[np.isnan(ls2) | np.isinf(ls2)] = 0
        cost = - (1 / m) * np.nansum(y_ * ls1 + (1-y_) * ls2)
        return cost
    
    @staticmethod
    def __cost_derivative_mlm(sigma_, y_):
        m = y_.shape[0]
        sigma_[(sigma_ == 1) | (sigma_ == 0)] = np.nan
        ret = (sigma_ - y_) / (sigma_ * (1 - sigma_))
        ret[np.isnan(ret)] = 0
        return ret / m

    @staticmethod
    def __cost_corr(sigma_, y_):
        size = y_.shape[0]
        ybar = np.mean(y_)
        sbar = np.mean(sigma_)
        cov = np.mean(sigma_ * (y_ - ybar))
        A = cov**2
        B = np.mean((sigma_ - sbar)**2)
        C = np.mean((y_ - ybar)**2)
        r2 = A / (B * C)
        return (1 - r2)

    @staticmethod
    def __cost_derivative_corr(sigma_, y_):
        size = y_.shape[0]
        ybar = np.mean(y_)
        sbar = np.mean(sigma_)
        cov = np.mean(sigma_ * (y_ - ybar))
        A = cov**2
        B = np.mean((sigma_ - sbar)**2)
        C = np.mean((y_ - ybar)**2)
        tmp = np.mean(sigma_ * (y_ - ybar))
        pA = (2 / size) * tmp * (y_ - ybar) 
        pB = (2 / size) * (sigma_ - sbar)
        return - (1 / ((B**2) * C)) * (pA * B - A * pB)

    # --- 10. regularization ---
    def __L2_cost(self, data_size_):
        O_sum=0
        for l in range(self.L - 1):
            O_local = self.O[l].copy()
            O_local[0, :] = 0
            if l < self.L - 2:
                O_local[:, 0] = 0
            O_sum += np.nansum(O_local ** 2)
        return (1 / data_size_) * (self.lambd / 2) * O_sum

    def __L2_derivative(self, data_size_):
        L2_dev = list()
        for l in range(self.L - 1):
            L2_dev.append((self.lambd / data_size_) * self.O[l])
            L2_dev[l][0, :] = 0
            if l < self.L - 2:
                self.O[l][:, 0] = 0
        L2_dev.append(None)
        return L2_dev


    # ==================== Appendex. Gradient Checking ====================
    def __Forward_GD_check(self, actfun_hidden_, actfun_output_, O_):
        for l in range(1, self.L - 1):
            self.Z[l] = self.F[l-1].dot(O_[l-1])
            self.F[l] = actfun_hidden_(self.Z[l])
            self.F[l][:, 0] = 1
            self.F[l] = self.F[l] * self.D[l]
            if self.dropout_probs:
                self.F[l][:, 1:] /= self.dropout_probs[l]
        self.Z[-1] = self.F[-2].dot(self.O[-2])
        self.F[-1] = actfun_output_(self.Z[-1]) 
    
    def __cost_GD_check(self, O_):
        self.__Forward_GD_check(self.actfun_hidden, self.actfun_output, O_)
        return self.costfun(self.F[-1], self.yr)
    
    def __O_copy(self):
        Oc = list()
        for l in range(self.L - 1):
            Oc.append(self.O[l].copy())
        Oc.append(None)
        return Oc
    
    def __GD_check(self, epsilon_=1e-7):
        Gc = list()
        for l in range(self.L - 1):
            Gc.append(self.G[l].copy())
        Gc.append(None)
        dQab = 0
        dQa  = 0
        dQb  = 0
        for l in range(self.L - 1):
            ni = Gc[l].shape[0]
            nj = Gc[l].shape[1]
            for i in range(ni):
                for j in range(1, nj):
                    Op = self.__O_copy()
                    Op[l][i, j] += epsilon_
                    cost_p = self.__cost_GD_check(Op)
                    Om = self.__O_copy()
                    Om[l][i, j] -= epsilon_
                    cost_m = self.__cost_GD_check(Om)
                    Gc[l][i, j] = (cost_p - cost_m) / (2 * epsilon_)
                    dQa  += np.sqrt(self.G[l][i,j] ** 2)
                    dQb  += np.sqrt(Gc[l][i,j] ** 2)
                    dQab += np.sqrt((self.G[l][i,j] - Gc[l][i,j]) ** 2)
        return dQab / (dQa + dQb)

        
def main():
    sdl = DeepLearning(nn_sizes_list_=[2, 7, 5, 3, 1], act_hidden_='relu', epochs_ = 235)
    np.random.seed(1)
    X = np.random.randn(2, 3).T
    y = np.array([[1.74481176], [-0.7612069], [0.3190391]])
    sdl.fit(X, y)
    sdl.plot_cost()
    plt.show()
    y_pred = sdl.predict(X)
    
    
    
if __name__ == '__main__':
    main()
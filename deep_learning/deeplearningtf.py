import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class DeepLearningTF:
    def __init__(self, nn_sizes_list_=None, act_hidden_='tanh', act_output_='sigmoid', 
                 cost_='mlm', optimizer_='GD', epochs_=1000, batch_size_=None, eta_=0.0075, 
                 lambd_=None, steps_cost_=100, keep_prob_=None):
        # --- input attributes ---
        self.nn = nn_sizes_list_
        self.act_hidden = act_hidden_
        self.act_output = act_output_
        self.cost_mode = cost_
        self.optimizer = optimizer_
        self.epochs = epochs_
        self.batch_size = batch_size_
        self.eta = eta_
        self.lambd = lambd_     
        self.steps_cost = steps_cost_
        self.kp = keep_prob_
        # --- class attributes ---
        self.session = None
        self.L = len(self.nn)
        self.X_tf = None   # X place holder
        self.Y_tf = None   # Y place holder
        self.W_tf = None   # Weights
        self.b_tf = None   # bias
        self.Z_tf = None   # X dot O + b
        self.F_tf = None   # sigma(Z)
        self.C_tf = None   # cost
        self.G_tf = None   # gradient descent
        self.kp_tf = None  # keep probability
        self.actfun_hidden = None
        self.actfun_output = None
        self.costs = None
        
    # ==================== I. Build-up a Neural Netwrok ====================
    def __plot_graphs(self):
        self.__plot_input()
        self.__plot_layers()
        self.__plot_cost()
        self.__plot_optimizer()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def __plot_input(self):
        self.X_tf = tf.placeholder(tf.float64, [None, self.nn[0]])
        self.Y_tf = tf.placeholder(tf.float64, [None, self.nn[-1]])
        if self.kp is not None:
            self.kp_tf = tf.placeholder(tf.float64)
        self.W_tf = list()
        self.b_tf = list()
        self.Z_tf = [None]
        self.F_tf = [self.X_tf]
    
    def __plot_layers(self):
        # --- 1. input layer ---
        W0_tf = tf.Variable(np.random.randn(self.nn[0], self.nn[1]) * np.sqrt(2 / self.nn[0]))
        b0_tf = tf.Variable(np.zeros((1, self.nn[1])))
        Z1_tf = tf.matmul(self.X_tf, W0_tf) + b0_tf
        F1_tf = self.actfun_hidden(Z1_tf)
        if self.kp_tf is not None:
            F1_tf = tf.nn.dropout(F1_tf, self.kp_tf)
        self.W_tf.append(W0_tf)
        self.b_tf.append(b0_tf)
        self.Z_tf.append(Z1_tf)
        self.F_tf.append(F1_tf)       
        # --- 2. hidden layers ---
        for l in range(1, self.L-1, 1):
            Wl_tf = tf.Variable(np.random.randn(self.nn[l], self.nn[l+1]) * np.sqrt(2 / self.nn[l]))
            bl_tf = tf.Variable(np.zeros((1, self.nn[l+1])))
            Zlp1_tf = tf.matmul(self.F_tf[l], Wl_tf) + bl_tf
            if l == self.L-2:
                Flp1_tf = self.actfun_output(Zlp1_tf)
            else:
                Flp1_tf = self.actfun_hidden(Zlp1_tf)
                if self.kp_tf is not None:
                    Flp1_tf = tf.nn.dropout(Flp1_tf, self.kp_tf)
            self.W_tf.append(Wl_tf)
            self.b_tf.append(bl_tf)
            self.Z_tf.append(Zlp1_tf)
            self.F_tf.append(Flp1_tf) 
        # --- 3. output layers ---
        l = self.L-1
        self.W_tf.append(None)
        self.b_tf.append(None)
    
    def __plot_cost(self):
        if self.cost_mode == 'mlm':
            self.C_tf = tf.reduce_mean((self.Y_tf - self.F_tf[-1])**2)
        elif self.cost_mode == 'binary':
            self.C_tf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y_tf, logits=self.F_tf[-1]))
        elif self.cost_mode == 'softmax':
            self.C_tf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_tf, logits=self.F_tf[-1]))
        else:
            raise ValueError('{} cost_mode is not available'.format(self.cost_mode))
        # add l2 regularization
        if self.lambd is not None:
            for l in range(self.L-1):
                self.C_tf += self.lambd * tf.nn.l2_loss(self.W_tf[l])
    
    def __plot_optimizer(self):
        if self.optimizer == 'GD':
            optimizer = tf.train.GradientDescentOptimizer(self.eta)
        elif self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(self.eta)
        else:
            raise ValueError('{} optimizer is not available'.format(self.optimizer))
        self.G_tf = optimizer.minimize(self.C_tf)
    
    # ==================== III. Gradient Descent ====================   
    def __minibatch(self, X_, Y_):
        np.random.seed(1)
        data_size = X_.shape[0]
        if self.batch_size is None:
            self.batch_size = data_size
        m = min(data_size, self.batch_size)
        for i in range(self.epochs + 1):
            shuffled_indices = np.random.permutation(data_size)
            X_shuffled = X_[shuffled_indices]
            y_shuffled = Y_[shuffled_indices]
            for r in range(0, data_size, self.batch_size):
                xr = X_shuffled[r:r + self.batch_size]
                yr = y_shuffled[r:r + self.batch_size]
                if self.kp is not None:
                    self.sess.run(self.G_tf, feed_dict={self.X_tf: xr, self.Y_tf: yr, self.kp_tf: self.kp})
                else:
                    self.sess.run(self.G_tf, feed_dict={self.X_tf: xr, self.Y_tf: yr})
                iterations = (r + i * data_size) // m
                if iterations % self.steps_cost == 0:
                    if self.kp is not None:
                        cost = self.sess.run(self.C_tf, feed_dict={self.X_tf: xr, self.Y_tf: yr, self.kp_tf: self.kp})
                    else:
                        cost = self.sess.run(self.C_tf, feed_dict={self.X_tf: xr, self.Y_tf: yr})
                    self.costs.append(cost)
                    print('epoch={}, cost={}'.format(i, cost))  
    
        
    # ==================== III. Initialization Before Fitting ====================   
    def __init_hidden(self):
        if self.act_hidden == 'tanh':
            self.actfun_hidden = tf.nn.tanh
        elif self.act_hidden == 'sigmoid':
            self.actfun_hidden = tf.nn.sigmoid
        elif self.act_hidden == 'relu':
            self.actfun_hidden = tf.nn.relu
        else:
            raise ValueError('{} is not available hidden activation function'.format(self.act_hidden))
        
    def __init_output(self):
        if self.act_output == 'tanh':
            self.actfun_output = tf.nn.tanh
        elif self.act_output == 'sigmoid':
            self.actfun_output = tf.nn.sigmoid
        elif self.act_output == 'relu':
            self.actfun_output = tf.nn.relu
        else:
            raise ValueError('{} is not available output activation function'.format(self.act_output))

    def __initializaiton(self):
        np.random.seed(3)
        self.__init_hidden()
        self.__init_output()
        self.costs = list()


    # ==================== IV. Fitting Function ====================
    def fit(self, X_, y_):
        self.__initializaiton()
        self.__plot_graphs()
        self.__minibatch(X_, y_)
        
    def predict(self, X_):
        if self.kp is not None:
            F = self.sess.run(self.F_tf[-1], feed_dict={self.X_tf: X_, self.kp_tf: 1.0})
        else:
            F = self.sess.run(self.F_tf[-1], feed_dict={self.X_tf: X_})
        return F
      
        
    # ==================== VI. Plot Functions ====================
    def plot_cost(self):
        iters = range(len(self.costs))
        plt.figure(figsize=(10,8))
        plt.plot(iters, self.costs)
        plt.xlabel("iterations (per {} steps)".format(self.steps_cost), fontsize=20)
        plt.ylabel("cost", fontsize=20)
        plt.title('learning rate = {}'.format(self.eta), fontsize=20)

        
        
        
        
        
        
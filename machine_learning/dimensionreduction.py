import pandas as pd
from sklearn import decomposition

class PCR:
    def __init__(self, m_, reg_model_):
        self.m = m_                   # number of priciple components
        self.pca = None               # pca instance in sklearn
        self.reg_model = reg_model_   # regression model we used for PCA
        self.scale_mean_std = None    # mean and std for scaling predictors
        self.X_size = None            # size of columns
        
    def fit(self, X_, y_):
        # === 1. scaled the predictors ===
        self.scale_mean_std = dict()
        self.X_size = X_.shape[1]
        colnames = [i for i in range(self.X_size)]
        df = pd.DataFrame(X_, columns=colnames)
        for i in range(self.X_size):
            mean = df[i].mean()
            std  = df[i].std()
            df[i] = (df[i]-mean)/std 
            self.scale_mean_std[i] = [mean, std]
        X_scaled = df.values
        # === 2. find the PCs from the scaled X_ ===
        self.pca = decomposition.PCA(n_components=self.m)
        self.pca.fit(X_scaled)
        Xpc = self.pca.transform(X_scaled)
        # === 3. fit model with Xpc ===
        self.reg_model.fit(Xpc, y_)
    
    def predict(self, X_):
        # === 1. scaled the predictors ===
        size = X_.shape[1]
        if size != self.X_size:
            raise ValueError('The column size of input X is {}; the size of training X is {}'
                             .format(size, self.X_size))
        colnames = [i for i in range(self.X_size)]
        df = pd.DataFrame(X_, columns=colnames)
        for i in range(self.X_size):
            mean = self.scale_mean_std[i][0]
            std  = self.scale_mean_std[i][1]
            df[i] = (df[i]-mean)/std 
        # === 2. make a prediction from the scaled X_ ===
        X_scaled = df.values
        Xpc = self.pca.transform(X_scaled)
        return self.reg_model.predict(Xpc)
    
    

def main():
    pass

if __name__ == "__main__":
    main()
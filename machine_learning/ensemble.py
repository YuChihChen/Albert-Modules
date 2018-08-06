import numpy as np
import pandas as pd

class Bagging:
    def __init__(self, reg_model_list_, mode_='bootstrap'):
        self.reg_model_list = reg_model_list_ 
        self.mode = mode_
        self.X_size = None
        
    def __fit_from_bootstrap_samples(self, model_, X_, y_):
        self.X_size = X_.shape[1]
        colnames = [i for i in range(self.X_size)]
        df = pd.DataFrame(X_, columns=colnames)
        df['y'] = y_
        df_shuffle = df.sample(frac=1, replace=True)
        X_train = df_shuffle.loc[:, colnames].values
        y_train = df_shuffle.loc[:, 'y'].values
        model_.fit(X_train, y_train)
        
    def fit(self, X_, y_):
        for reg_model in self.reg_model_list:
            if self.mode == 'bootstrap':
                self.__fit_from_bootstrap_samples(reg_model, X_, y_)
            else:
                raise ValueError('{} mode is not avaliable'.format(self.mode))
    
    def predict(self, X_):
        vec = np.array([None]*X_.shape[0])
        y_df = pd.DataFrame(vec.T, columns=['drop'])
        for i in range(len(self.reg_model_list)):
            reg_model = self.reg_model_list[i]
            y_pred = reg_model.predict(X_)
            y_df['{}'.format(i)] = y_pred
        y_df.drop(columns=['drop'], inplace=True)
        return y_df.mean(axis=1).values


class ResidualBoosting:
    def __init__(self, reg_model_list_):
        self.reg_model_list = reg_model_list_ 
        self.X_size = None
    
    def fit(self, X_, y_):
        res = y_
        y_pred = 0
        for reg_model in self.reg_model_list:
            res = res-y_pred
            reg_model.fit(X_, res)
            y_pred = reg_model.predict(X_)
    
    def predict(self, X_):
        vec = np.array([None]*X_.shape[0])
        y_df = pd.DataFrame(vec.T, columns=['drop'])
        for i in range(len(self.reg_model_list)):
            reg_model = self.reg_model_list[i]
            y_pred = reg_model.predict(X_)
            y_df['{}'.format(i)] = y_pred
        y_df.drop(columns=['drop'], inplace=True)
        return y_df.sum(axis=1).values
    
    

def main():
    pass

if __name__ == "__main__":
    main()
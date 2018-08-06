import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CrossValidation:
    def __init__(self, df_):
        self.df = df_.copy()
        self.df_scaled = None
        self.train_test_dfs = None
        
    def scale_predictors(self, response_, mode_='standarlized'):
        self.df_scaled = self.df.copy()
        cols = list(self.df.columns)
        cols.remove(response_)
        if mode_ == 'standarlized':
            scaled_funtion = lambda x: (x-x.mean())/x.std()
        else:
            raise ValueError('{} mode is not avaliable for scaling'.format(mode_))
        self.df_scaled = self.df_scaled[cols].apply(scaled_funtion)
        self.df_scaled[response_] = self.df[response_]
        self.df_scaled = self.df_scaled[[response_]+cols]

    def train_test_split(self, k_folds_, scale_=True):
        if scale_ is False:
            self.df_scaled= self.df
        if self.df_scaled is None:
            raise ValueError('df_scaled is empty, apply "scale_predictors"')
        print("train_test split with scaled predictors: {}".format(scale_))
        # shuffle the data and label them by folds
        df_shuffle = self.df_scaled.sample(frac=1, replace=False)
        bin_size = len(df_shuffle.index)//k_folds_
        label = list()
        for k in range(k_folds_-1):
            label.extend([k]*bin_size)
        label.extend([k_folds_-1]*(len(df_shuffle.index)-len(label)))
        df_shuffle['label'] = label
        # 3. conduct cross-validation
        train_test_list = list()
        for k in range(k_folds_):
            df_test  = df_shuffle[df_shuffle.label == k].drop(columns=['label'])
            df_train = df_shuffle[~df_shuffle.index.isin(df_test.index)].drop(columns=['label'])
            train_test_list.append([df_train, df_test])
        self.train_test_dfs = train_test_list
        return train_test_list
    
    def __cv_get_a_test_error(self, model_, response_):
        """
        model_: the model in sklearn module
        response_: the name of 'y'
        return: estimated test error and its stadnard error
        """
        if self.train_test_dfs is None:
            raise ValueError('train_test_dfs is empty, apply "train_test_split" first')
        # get the columns of predictors
        self.df_scaled = self.df.copy()
        cols = list(self.df.columns)
        cols.remove(response_)
        # get the estimated testing error and error
        errors = list()
        for train_test_df in self.train_test_dfs:
            X_train = train_test_df[0][cols].values
            y_train = train_test_df[0][response_].values
            X_test  = train_test_df[1][cols].values
            y_test  = train_test_df[1][response_].values
            model_.fit(X_train, y_train)
            y_pred  = model_.predict(X_test)
            error = np.mean((y_test-y_pred)**2)
            errors.append(error)    
        return np.mean(errors), np.std(errors)
    
    def __cv_plot_test_errors(self, lambdas_):
        if self.test_error_means is None:
            raise ValueError('test_error_means is empty, apply "cv_get_test_errors" first')
        plt.errorbar(lambdas_, self.test_error_means, yerr=self.test_error_stds,
                     ecolor='r', marker='.', ms=10, ls='--')
        plt.title('Error vs. Lambda')
        plt.xlabel('lambda')
        plt.ylabel('error')
        plt.show()
    
    def cv_get_test_errors(self, models_, response_, plot_=False, lambdas_=None):
        self.test_error_means = list()
        self.test_error_stds  = list()
        for model in models_:
           mean, std = self.__cv_get_a_test_error(model, response_)
           self.test_error_means.append(mean)
           self.test_error_stds.append(std)
        if plot_ == True:
            if (lambdas_ is None) or (len(lambdas_) != len(self.test_error_means)):
                raise ValueError('lambda is None or len(lambda) != len(models)')
            self.__cv_plot_test_errors(lambdas_)
        return self.test_error_means, self.test_error_stds

def main():
    # ====== data from simulation ======
    n = 500
    x1 = np.random.normal(loc=1 , scale=np.sqrt( 1), size=n)
    x2 = np.random.normal(loc=5 , scale=np.sqrt( 5), size=n)
    x3 = np.random.normal(loc=10, scale=np.sqrt(10), size=n)
    e  = np.random.normal(loc=0 , scale=np.sqrt( 2), size=n)
    y  = 2 + x1 + x2 + e
    df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3})
    print(df.head())
    # ====== scaled the data ======
    cv = CrossValidation(df)
    cv.scale_predictors('y')
    train_test_dfs = cv.train_test_split(5)
    print(train_test_dfs[4][0].shape)
    print(train_test_dfs[4][1].shape)

    plt.errorbar([0,1,2,3], [0,1,2,3], [0,1,2,3], ecolor='r', marker='.', ms=10, ls='--')

if __name__ == "__main__":
    main()
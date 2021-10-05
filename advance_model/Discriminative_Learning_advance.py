#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from training_model.discriminative_learning import Discriminative_Learning


# In[4]:


class DL_advance:
    
    def __init__(self, model_class):
        self.model = model_class()


    def DL_init(self, model_info = {}):
        '''
            Initiate value for parameters
            path_csv: is path to csv database file
            epsilon for stop condition, L2(Wk+1 - W)
            thresshold for convert from P(Y=1|X) to Y estimate
            rate is propotion of training in database. EX: rate = 0.8, training = 80%, test = 20%
        '''
        self.model.init(model_info)
        
    def fit_adv(self):
        '''
            try to improve stop criteria by using accuracy of training & validation data
            with each trained W, accuracy of training, test are calculated.
            after all, find out which W with highest test accuracy & difference between training-test is small.
            
            X, Y are input including both Training & Validation
            X [m, n_train + n_valid]
            Y [1, n_train + n_valid]
    
        
            learning_rate = type1 -> alpha = 1/ (k+1) with k is iteration

        '''
        
        # import base functions
        
        # load input file
        data_table = self.model.load_input()
        
        database = data_table.T
        
        row, col = np.shape(database)
        
        X = database[0:row-1, :]
        Y = database[row-1, :]
        
        
        # Insert feature x0 = 1 to input X
        m, n = np.shape(X)
        X_1 = np.ones([m+1, n], dtype = np.float64)
        X_1[0:m, :] = X
        X = X_1
        
        #normalization for input feature
        #feature had been normalized already
        
        m, n = np.shape(X) # m is number of feature, n is number of observation 
        
        # make permutation for database
        seed = 0
        np.random.seed(seed)
        idx_per = np.random.permutation(n)
        Y_per = Y[idx_per].reshape(1, n)
        X_per = X[:, idx_per]  
        
        
        # separate training set and validation set
        # take from index 0 to 80% of input X, Y as Training, otherwise for Validation
        temp = int(self.model.rate * n) 
        X_train = X_per[:,0:temp+1]
        X_test = X_per[:, temp+1 : ]
        
        Y_train = Y_per[:, 0:temp+1]
        Y_test = Y_per[:, temp+1 : ]
        
        
        W = np.random.randn(m,1)
        W_pre = np.zeros(np.shape(W))
    
        W_vec = []
        Acc_test_vec = []
        Acc_train_vec = []
    
        # iteration for training Weight
        for idx in range (4000):     
    
            W = W + (1.0 / (idx+1)) * np.sum( X_train * (Y_train - self.model.sigmoid(W.T @ X_train)), axis = 1, keepdims = True)
    
            # stop condition 
            # check Accuracy of Training sample and Validation Sample
            # make sure not become high Bias & Variation        
            Y_train_est = self.model.predict(X_train[:-1,:], W)
            Y_test_est = self.model.predict( X_test[:-1,:], W)
            
            acc_train = self.model.Accu_eval( Y_train_est, Y_train)
            acc_test = self.model.Accu_eval( Y_test_est, Y_test)
            
            # visualization delta
            Acc_train_vec.append(acc_train)
            Acc_test_vec.append(acc_test)
            W_vec.append(W)
 
        # stop criteria
        Acc_train_arr = np.array(Acc_train_vec)
        Acc_test_arr = np.array(Acc_test_vec)
        W_arr = np.array(W_vec)
        idx = np.argmax(Acc_test_arr[np.abs(Acc_train_arr-Acc_test_arr)<0.03])
            
        # plot figure of accuracy training/test
        x_lab = [x_lab for x_lab in range (len(Acc_train_vec))]
        plt.plot(x_lab, Acc_train_vec, color = 'r', label = 'training accuracy')
        plt.plot(x_lab, Acc_test_vec, color = 'b', label = 'test accuracy')
        plt.axvline(x =idx, color='g', linestyle='-')
        plt.title('accuracy of database _{}'.format(self.model.path_csv))
        plt.legend(loc = 'upper right')
        plt.legend()
        plt.show()
            
        return W_arr[idx,:,0], Acc_train_vec[idx], Acc_test_vec[idx]
          
            
    def training_feature(self, X, Y):
        '''
            training model with input removed some feature 
            It same with training_model() but take input instead of load from .csv file
        '''
        
        #normalization for input feature
        #feature had been normalized already
        
        m, n = np.shape(X) # no of feature, n is no of observation 
        
        
        
        # make permutation for database
        seed = 0
        np.random.seed(seed)
        idx_per = np.random.permutation(n)
        Y_per = Y[idx_per].reshape(1, n)
        X_per = X[:, idx_per]  
        
        
        # separate training set and validation set
        # take from index 0 to 80% of input X, Y as Training, otherwise for Validation
        temp = int(self.model.rate * n) 
        X_train = X_per[:,0:temp+1]
        X_test = X_per[:, temp+1 : ]
        
        Y_train = Y_per[:, 0:temp+1]
        Y_test = Y_per[:, temp+1 : ]                
        

        # decleare hyperparameter
        W = np.random.randn(m+1,1)
        
        # evaluate model
        W_trained = self.model.fit(X_train, Y_train, self.model.learning_type, self.model.alpha, self.model.epsilon)
        Y_train_est = self.model.predict(X_train, W_trained)
        Y_test_est = self.model.predict( X_test, W_trained)
            
        acc_train = self.model.Accu_eval( Y_train_est, Y_train)
        acc_test = self.model.Accu_eval( Y_test_est, Y_test)
    
        return acc_train, acc_test

    def remove_feature(self):
        '''
            this function is to remove as much feature as posible.
            step_wise search algorithm.
            step1: training for the first feature. 
            step2: add new more feature and training. if accuracy increase -> keep that new feature, 
                                                      else remove new feature
            do step 2 until all feature have been tested.
        '''
        # load input file
        data_table = self.model.load_input()
        
        database = data_table.T
        
        row, col = np.shape(database)
        
        # training with only 1 feature
        X = database[0, :].reshape(1, col)
        Y = database[row-1, :]
        print(np.shape(X))
        acc_train_prev, acc_test_prev = self.training_feature(X, Y)
        
        idx_feature = [1]
        for i in range(1, row-1):
            New_fea = database[i, :].reshape(1, col)
            X_cur = np.concatenate((X, New_fea), axis = 0)
            acc_train_cur, acc_test_cur = self.training_feature(X_cur, Y)
            if (np.abs(acc_train_cur - acc_test_cur) < 0.05) and acc_test_cur > acc_test_prev:
                acc_test_prev = acc_test_cur
                acc_train_prev = acc_train_cur
                X = X_cur
                idx_feature.append(i+1)
            else:
                pass
            
            # Save remaining feature to .csv file
            #new_data = np.concatenate((X, Y.reshape(1, col)), axis = 0).T
            #np.savetxt('new_feature.csv'.format(self.model.path_csv), new_data, delimiter=',')
            
        return acc_train_prev, acc_test_prev, idx_feature


#=================================================================================
    def add_feature(self):
        '''
            this function is to add new feature created by no-linear function of existed feature
            x1.x2, x1.x3, .. x1.xend will be check.
        '''
        # load input file
        data_table = self.model.load_input()
        
        database = data_table.T
        
        row, col = np.shape(database)
        
        # training with only 1 feature
        X = database[0:row-1, :]
        Y = database[row-1, :]
        #print(np.shape(X))
        #acc_train_prev, acc_test_prev = self.training_feature(X, Y)
        
        acc_train_prev = 0
        acc_test_prev = 0
        idx_feature = []
        acc_train_vec = []
        acc_test_vec = []
        
        X_org = X
        
        # check for adding new features xi*xj
        for i in range(0, row-1):
            New_fea = X_org[i-1, :].reshape(1, col) * X_org[i, :].reshape(1, col)
            X_cur = np.concatenate((X, New_fea), axis = 0)
            acc_train_cur, acc_test_cur = self.training_feature(X_cur, Y)
            
            # if new feature have good accuraccy, update X
            if (np.abs(acc_train_cur - acc_test_cur) < 0.05) and acc_test_cur > acc_test_prev:
                acc_test_prev = acc_test_cur
                acc_train_prev = acc_train_cur
                X = X_cur
                idx_feature.append([i-1, i])
                acc_train_vec.append(acc_train_prev)
                acc_test_vec.append(acc_test_prev)
            else:
                pass
            
        # check for adding new features xi**2    
        for i in range(0, row-1):

            New_fea = np.square(X_org[i-1, :].reshape(1, col))
            X_cur = np.concatenate((X, New_fea), axis = 0)
            acc_train_cur, acc_test_cur = self.training_feature(X_cur, Y)
            
            # if new feature have good accuraccy, update X
            if (np.abs(acc_train_cur - acc_test_cur) < 0.05) and acc_test_cur > acc_test_prev:
                acc_test_prev = acc_test_cur
                acc_train_prev = acc_train_cur
                X = X_cur
                idx_feature.append([i-1, i-1])
                acc_train_vec.append(acc_train_prev)
                acc_test_vec.append(acc_test_prev)
            else:
                pass
            
            #Save remaining feature to .csv file
            #new_data = np.concatenate((X, Y.reshape(1, col)), axis = 0).T
            #np.savetxt('add_new_feature{}.csv'.format(self.model.path_csv), new_data, delimiter=',')
            
        return acc_train_vec, acc_test_vec, idx_feature




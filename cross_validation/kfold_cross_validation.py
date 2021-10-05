#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from discriminative_learning import Discriminative_Learning
from lda_learning import LDA_Learning


# In[3]:


class kfold_cross_validation:
    
    def __init__(self, model_class):
        self.model = model_class()


    def kflod_init(self, kfold_info = {}, model_info = {}):
        '''
            Initiate value for parameters
            path_csv: is path to csv database file
            epsilon for stop condition, L2(Wk+1 - W)
            thresshold for convert from P(Y=1|X) to Y estimate
            rate is propotion of training in database. EX: rate = 0.8, training = 80%, test = 20%
        '''
        
        self.kfold = kfold_info.get('kflod', 10)
        self.path_csv = kfold_info.get('path', 'red_wine.csv') 
        
        self.model.init(model_info)

    def kfold_load_data(self):
        '''
            input .csv file will be loaded and separate into each fold.
            kfold is number fold (self.kfold)
        '''
        # load input file
        data_table = self.model.load_input()
        
        database = data_table.T
        
        row, col = np.shape(database)
        
        X = database[0:row-1, :]
        Y = database[row-1, :]
        
        return X, Y
    
    def kfold_data_separate(self, X, Y):

       
        m, n = np.shape(X) # no of feature, n is no of observation 
                
        # make permutation for database
        seed = 0
        np.random.seed(seed)
        idx_per = np.random.permutation(n)
        Y_per = Y[idx_per].reshape(1, n)
        X_per = X[:, idx_per]  
        
        # break down data into kfold part, the last part may be longer than others.
        size_val = n // self.kfold 
        X_fold = {}
        Y_fold = {}
        for i in range(self.kfold-1):
            X_fold[str(i)] = X_per[:, i*size_val : (i+1)*size_val]
            Y_fold[str(i)] = Y_per[:, i*size_val : (i+1)*size_val].reshape(1, size_val) # 2D array
        
        # make sure the last fold is longer than others
        X_fold[str(self.kfold-1)] = X_per[:, (i+1)*size_val :] # [observation x feature]
        Y_fold[str(self.kfold-1)] = Y_per[:, (i+1)*size_val :].reshape(1, n-(self.kfold-1)*size_val)
        
        return X_fold, Y_fold
    
    def kflod_validation(self):
        '''
            xxx
        '''
        X, Y = self.kfold_load_data()
        X_fold, Y_fold = self.kfold_data_separate(X, Y)
        m, n_fold = np.shape(X_fold['0'])
        
        ACCU_TRAIN = []
        ACCU_VALID = []
        
        # simulate fold validation
        for iter_fold in range (self.kfold):
            X_train_pre = np.zeros((m, 1)) # redundant 0-array for same size when np.concatenate (1)
            Y_train_pre = np.zeros((1, 1))
        
            for j in range (self.kfold):
                if j == iter_fold:
                    X_valid = X_fold[str(j)]
                    Y_valid = Y_fold[str(j)]
                elif j != iter_fold:
                    X_train_pre = np.concatenate((X_train_pre, X_fold[str(j)]), axis = 1)
                    Y_train_pre = np.concatenate((Y_train_pre, Y_fold[str(j)]), axis = 1)
        
            X_train = X_train_pre[:,1:] #remove redundant 0-array from (1)
            Y_train = Y_train_pre[:,1:] #remove redundant 0-array

            # evaluate model
            if self.model.name == 'discriminative_learning':
                W_trained = self.model.fit(X_train, Y_train, self.model.learning_type, self.model.alpha, self.model.epsilon)
            elif self.model.name == 'lda_learning':
                W_trained = self.model.fit(X_train, Y_train)
            else:
                raise ValueError('wrong training model')
            
            Y_train_est = self.model.predict(X_train, W_trained)
            Y_valid_est = self.model.predict(X_valid, W_trained)
        
            acc_train = self.model.Accu_eval( Y_train_est, Y_train)
            acc_valid = self.model.Accu_eval( Y_valid_est, Y_valid)
        
            ACCU_TRAIN.append(acc_train)
            ACCU_VALID.append(acc_valid)
             
    
        return np.mean(ACCU_TRAIN), np.mean(ACCU_VALID)

    
    
kfold_info = {'kflod': 10, 'path': 'red_wine.csv'}

DL_info =    {'epsilon': 0.05,
              'learning_type': 'type1',
              'alpha': 0.2,
              'thresshold': 0.5,
              'rate': 0.8
          }


DL_model = Discriminative_Learning
model = kfold_cross_validation(DL_model)
model.kflod_init(kfold_info, DL_info)

DL_train_acc, DL_valid_acc = model.kflod_validation()
print('Discriminative learning training model')
print('The accuracy of training for {} database = '.format(kfold_info['path']) + str(DL_train_acc))
print('The accuracy of test for {} database = '.format(kfold_info['path']) + str(DL_valid_acc))


LDA_info = {  'thresshold': 0,
              'rate': 0.8
          }
LDA_model = LDA_Learning

model = kfold_cross_validation(LDA_model)
model.kflod_init(kfold_info, LDA_info)
LDA_train_acc, LDA_valid_acc = model.kflod_validation()
print('LDA training model')
print('The accuracy of training for {} database = '.format(kfold_info['path']) + str(LDA_train_acc))
print('The accuracy of test for {} database = '.format(kfold_info['path']) + str(LDA_valid_acc))



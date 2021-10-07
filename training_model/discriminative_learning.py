#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import time

class Discriminative_Learning:
    
    def init(self, DL_info = {}):
        '''
            Initiate value for parameters
            path: is folder of .csv files
            epsilon for stop condition, L2(Wk+1 - W)
            learning_type is 
                type1: 1/(k+1)
                type2: 1/2**t
                type3: fixed by alpha = 0.x
            thresshold for convert from P(Y=1|X) to Y estimate
            rate is propotion of training in database. EX: rate = 0.8, training = 80%, test = 20%
        '''
        self.path = DL_info.get('path', 'database')
        self.file = DL_info.get('file', 'red_wine.csv')

        self.epsilon = DL_info.get('epsilon', 0.05) 
        self.learning_type = DL_info.get('learning_type', 'type1')
        self.alpha = DL_info.get('alpha', 0.2)
        self.thresshold = DL_info.get('thresshold', 0.5)
        self.rate = DL_info.get('rate', 0.8)
        self.name = 'discriminative_learning'


    def load_input(self):
        '''
            path_csv is path of input.csv file
            EX: red_wine.csv, diabetes.csv
                diabet = np.genfromtxt('diabetes.csv', delimiter=",")
        '''
        path_csv = self.path + '/' + self.file
        trainig_data = np.genfromtxt(path_csv, delimiter=",")
        
        return trainig_data

    def fig_histogram(self):
        '''
            This function take a database with shape of pandas
            Return histogram figure for each feature of database (bar() and hist())
        '''
        database = self.load_input()
        group0 = database[database[:, -1]==0] # group all observation with label = 0
        group1 = database[database[:, -1]==1] # group all observation with label = 1
        
        x_lab = [x_lab for x_lab in range (np.shape(database)[0])]
        for idx in range (np.shape(database)[1]-1):
            fig = plt.figure(figsize=(10,10))
            # histogram of both group0 and group1
            ax = fig.add_subplot(311)
            ax.hist(database[:,idx], bins=20, label="group0&1")
            ax.legend(loc = 'upper right')
            ax.legend()
            ax.title.set_text('Feature_{} / Histogram of database'.format(idx+1))
            
            #histogram of each group
            ax = fig.add_subplot(312)
            ax.hist(group0[:,idx], bins=100, alpha=0.5, label="group0")
            ax.hist(group1[:,idx], bins=100, alpha=0.5, label="group1")
            ax.legend(loc = 'upper right')
            ax.legend()
            ax.title.set_text('Feature_{} / Histogram of Group0 and Group1'.format(idx+1))
            
            #plot of value of database
            ax = fig.add_subplot(313)
            ax.bar(x_lab, database[:,idx])
            ax.title.set_text('Feature_{} / Each sample in database'.format(idx+1))
            plt.show()
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, Y, learning_type, alpha, epsilon):
        '''
            X, Y are training input with m features, n observation
            X [m, n]
            Y [1, n]

        
            learning_rate = type1 -> alpha = 1/ (k+1) with k is iteration
                            type2 -> alpha = 1/2 ** t
                            type3 -> fixed alpha
            epsilon is used as stop training condition
            
            return: trained Weight
        '''
        
        assert epsilon > 0
        assert np.shape(X)[1] == np.shape(Y)[1]
        
        # Insert feature x0 = 1 to input X
        m, n = np.shape(X)
        X_train = np.ones([m+1, n], dtype = np.float64)
        X_train[0:m, :] = X

        Y_train = Y.reshape(1, n)
        
        # Weight parameter initialize 
        W = np.random.randn(m+1,1)
        
        W_pre = np.zeros(np.shape(W))
    
        W_vec = []
    
        delta = float('inf')
        # iteration for training Weight
        idx = 0
        while (delta >= epsilon):
            
            if learning_type == 'type1':
                W = W + (1.0 / (idx+1)) * np.sum( X_train * (Y_train - self.sigmoid(W.T @ X_train)), axis = 1, keepdims = True)
            elif learning_type == 'type2':
                W = W + 0.5**(idx // 100 + 1) * np.sum( X_train * (Y_train - self.sigmoid(W.T @ X_train)), axis = 1, keepdims = True)
            elif learning_type == 'type3':
                W = W + alpha * np.sum( X_train * (Y_train - self.sigmoid(W.T @ X_train)), axis = 1, keepdims = True)
            else:
                raise ValueError('It is not training type')
            
            # stop condition         
            delta = np.sum(np.square(W - W_pre))
            #loss = corss_entropy_loss(X, Y, W) # stop conditon by loss function
            W_pre = W
            idx += 1
            # visualization delta
            W_vec.append(W)

        return W
    
    def predict(self, X, W_trained):
        '''
            This function calculate estimated Y output
        '''
        
        # Insert feature x0 = 1 to test data X
        m, n = np.shape(X)
        X_test = np.ones([m+1, n], dtype = np.float64)
        X_test[0:m, :] = X
        
        # calculate percentage of P(Y=1|X)
        PY_1 = self.sigmoid ( W_trained.T @ X_test )
        
        # Convert to Y by compare with thresshold.
        # if >thresshold -> Y = 1, else Y = 0
        Y_est = (PY_1 >= self.thresshold) * 1.0         
        
        return Y_est

    def Accu_eval(self, Y_est, Y_test):
        '''
            compare estimated Y and testing Y
            return accuracy level
        '''
        
        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(len(Y_est.T)):
            if Y_est[0,i] == 1 and Y_test[0,i] == 1 :
                TP += 1
            elif Y_est[0,i] == 0 and Y_test[0, i] == 0 :
                TN += 1            
            elif Y_est[0, i] == 0 and Y_test[0, i] == 1 :
                FN += 1            
            elif Y_est[0, i] == 1 and Y_test[0, i] == 0 :
                FP += 1
            else:
                raise ValueError('predict() error') 
        
        Accuracy = (TP + TN) * 1.0 / (TP + TN + FP + FN)
        
        
        return Accuracy


    def training_model(self):
        '''
            training model using input from path_data
            rate is percentage of training in database
        '''
        # load input file
        data_table = self.load_input()
        
        database = data_table.T
        
        row, col = np.shape(database)
        
        X = database[0:row-1, :]
        Y = database[row-1, :]
        
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
        temp = int(self.rate * n) 
        X_train = X_per[:,0:temp+1]
        X_test = X_per[:, temp+1 : ]
        
        Y_train = Y_per[:, 0:temp+1]
        Y_test = Y_per[:, temp+1 : ]
        

        # decleare hyperparameter
        W = np.random.randn(m+1,1)
        
        # evaluate model
        W_trained = self.fit(X_train, Y_train, self.learning_type, self.alpha, self.epsilon)
        Y_train_est = self.predict(X_train, W_trained)
        Y_test_est = self.predict( X_test, W_trained)
            
        acc_train = self.Accu_eval( Y_train_est, Y_train)
        acc_test = self.Accu_eval( Y_test_est, Y_test)
    
        return acc_train, acc_test

    def comp_learning_rate(self):


        #self.init() # Initiate with default value
        
        # decleare hyperparameter

        #learning_type_vec = ['type1','type2','type3']
        learning_type_vec = ['type1','type2']
        alpha_vec = [0.2]

        
        acc_train_model = {}
        acc_valid_model = {}

            
        # check training time for each database 

        for learning_type in learning_type_vec:
            acc_train_model[learning_type] = {}
            acc_valid_model[learning_type] = {}
            if learning_type == 'type3':
                for alpha in alpha_vec:

                    start_time = time.time()
                    acc_train_model[learning_type][str(alpha)] = []
                    acc_valid_model[learning_type][str(alpha)] = []
                    
                    self.learning_type = learning_type
                    self.alpha = alpha
                    
                    # call training model
                    acc_train, acc_test = self.training_model()
                    
                    # save accurate value for each iteration
                    acc_train_model[learning_type][str(alpha)].append(acc_train)
                    acc_valid_model[learning_type][str(alpha)].append(acc_test)
                    print("---Training time of database {} with alpha = {} is %s seconds ---".format(learning_type, alpha) % (time.time() - start_time))
                    
            elif learning_type == 'type1' or learning_type == 'type2':
                    start_time = time.time()
                    acc_train_model[learning_type] = []
                    acc_valid_model[learning_type] = []
                    
                    self.learning_type = learning_type
                
                    # call training model
                    acc_train, acc_test = self.training_model()
                    
                    # save accurate value for each iteration
                    acc_train_model[learning_type].append(acc_train)
                    acc_valid_model[learning_type].append(acc_test)
                    print("---Training time of database {}  is %s seconds ---".format(learning_type) % (time.time() - start_time))
            else:
                raise ValueError('Wrong learning type, type1, type2, type3 are allow.')
            
            

        return acc_train_model, acc_valid_model
    

    def comp_thresshold(self):
        '''
            This function call training model with thresshold change inside [0, 1]
        '''
        
        acc_train_thres = []
        acc_test_thres = []
        thresshold_vec = np.linspace(0, 1, 30)
        for thresshold in thresshold_vec:
            self.thresshold = thresshold
            acc_train, acc_test = self.training_model()
            acc_train_thres.append(acc_train)
            acc_test_thres.append(acc_test)
        idx = [i for i in range(len(thresshold_vec))]
        plt.plot(thresshold_vec, acc_train_thres, color = 'r', label = 'train')
        plt.plot(thresshold_vec, acc_test_thres, color = 'b', label = 'test')
        plt.legend(loc = 'upper right')
        plt.legend()
        plt.grid()
        plt.xlabel('Thresshold')
        plt.ylabel('Accuracy')
        plt.title('Changing of Accuracy follow Thresshold of {}'.format(self.path + '/' + self.file))
        plt.show()
    
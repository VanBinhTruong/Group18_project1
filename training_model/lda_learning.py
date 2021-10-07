#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class LDA_Learning():
        
    def init(self, DL_info = {}):
        '''
            Initiate value for parameters
            path: is path to csv database file
            thresshold for convert from P(Y=1|X) to Y estimate
            rate is propotion of training in database. EX: rate = 0.8, training = 80%, test = 20%
        '''
        self.path = DL_info.get('path', 'database')
        self.file = DL_info.get('file', 'red_wine.csv')
        self.thresshold = DL_info.get('thresshold', 0)
        self.rate = DL_info.get('rate', 0.8)
        self.name = 'lda_learning'
    
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
        
    def cal_mu_sig(self, X0, X1):
        '''
            X0 is group of feature with lable 0
            X1 is group of feature with label 1
            return mu0, mu1, sigma 
        '''
        mu0 = np.array([np.mean(X0, axis = 1)]).T
        mu1 = np.array([np.mean(X1, axis = 1)]).T
        
        sig = ((X0-mu0) @ (X0-mu0).T + (X1-mu1) @ (X1-mu1).T)/(np.shape(X0)[1] + np.shape(X1)[1] -2)
        return mu0, mu1, sig

    def fit(self, X, Y):
        '''
            X, Y are training input with m features, n observation
            X [m, n]
            Y [1, n]

        
            mu0, mu1, sigma will be calculate by using cal_mu_sig() function
            
            return: trained Weight
        '''
        assert np.shape(X)[1] == np.shape(Y)[1]
        
        # separate group 0 and group 1
        m, n = np.shape(X)
        X_group_0 = []
        X_group_1 = []
        for i in range (n):
            if Y[0, i] == 1 :
                X_group_1.append(X[:, i])
            elif Y[0, i] == 0 :
                X_group_0.append(X[:, i])
            else:
                raise ValueError('Invalid value in Y training data')
                
        X_group_1 = np.array(X_group_1).T
        X_group_0 = np.array(X_group_0).T
        
        # calculate mu0, mu1, sigma
        mu0, mu1, sigma = self.cal_mu_sig(X_group_0, X_group_1)
        
        P_Y_1 = np.shape(X_group_1)[1] / n
        P_Y_0 = np.shape(X_group_0)[1] / n
        

        w0 = np.log(P_Y_1/P_Y_0) - 0.5 * (mu1.T @ np.linalg.inv(sigma) @ mu1) + 0.5 * (mu0.T @ np.linalg.inv(sigma) @ mu0) 
        
        w = np.linalg.inv(sigma) @ (mu1 - mu0) 
        
        # concate w0 into array w
        W_arr = np.concatenate((w, w0), axis = 0)
        
        
        return W_arr
        
    
    def predict(self, X, W_trained):
        '''
            This function calculate estimated Y output
        '''
        
        # Insert feature x0 = 1 to test data X
        m, n = np.shape(X)
        X_test = np.ones([m+1, n], dtype = np.float64)
        X_test[0:m, :] = X
        
        # calculate percentage of P(Y=1|X)
        
        log_odd_ratio = W_trained.T @ X_test 
        
        # Convert to Y by compare with thresshold.
        # thresshold = 0 in this training
        # if >thresshold -> Y = 1, else Y = 0
        Y_est = (log_odd_ratio >= self.thresshold) * 1.0
        
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


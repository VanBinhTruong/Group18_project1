#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from kfold_cross_validation.kfold_cross_validation import kfold_cross_validation
from utils.utils import split_train_test, accu_eval


class DL_advance:

    def __init__(self, kfold_class, model_class):
        self.kfold = kfold_class(model_class)

    def DL_init(self, kfold_info={}, model_info={}):
        '''
            Initiates the values for parameters
            path_csv: path to csv database file
            epsilon: stop condition, L2(Wk+1 - W)
            threshold: threshold to convert from P(Y=1|X) to Y estimate
            rate: proportion of training in database. EX: rate = 0.8, training = 80%, test = 20%
        '''
        self.kfold.kflod_init(kfold_info, model_info)

    def fit_adv(self):
        '''
            Tries to improve stop criteria by using accuracy of training & validation data
            Calculates the accuracy of training and test set for each trained W
            Selects W with the lowest bias and variance
            
            X, Y are input including both training & validation
            X [m, n_train + n_valid]
            Y [1, n_train + n_valid]

            learning_rate = type1 -> alpha = 1/(k+1) with k is iteration
        '''

        # load input file
        data_table = self.kfold.model.load_input()

        database = data_table.T

        row, col = np.shape(database)

        X = database[0:row - 1, :]
        Y = database[row - 1, :]

        # Insert feature x0 = 1 to input X
        m, n = np.shape(X)
        X_1 = np.ones([m + 1, n], dtype=np.float64)
        X_1[0:m, :] = X
        X = X_1

        X_train, X_test, Y_train, Y_test = split_train_test(X, Y, self.kfold.model.rate)

        W = np.random.randn(m, 1)

        W_vec = []
        Acc_test_vec = []
        Acc_train_vec = []

        # iteration for training Weight
        for idx in range(4000):
            W = W + (1.0 / (idx + 1)) * np.sum(X_train * (Y_train - self.kfold.model.sigmoid(W.T @ X_train)), axis=1,
                                               keepdims=True)

            # stop condition 
            # check accuracy of training sample and validation Sample
            # make sure they did not become high bias or high variation
            Y_train_estimate = self.kfold.model.predict(X_train[:-1, :], W)
            Y_test_estimate = self.kfold.model.predict(X_test[:-1, :], W)

            acc_train = accu_eval(Y_train_estimate, Y_train)
            acc_test = accu_eval(Y_test_estimate, Y_test)

            # visualization delta
            Acc_train_vec.append(acc_train)
            Acc_test_vec.append(acc_test)
            W_vec.append(W)

        # stop criteria
        Acc_train_arr = np.array(Acc_train_vec)
        Acc_test_arr = np.array(Acc_test_vec)
        W_arr = np.array(W_vec)
        idx = np.argmax(Acc_test_arr[np.abs(Acc_train_arr - Acc_test_arr) < 0.03])

        # plot figure of accuracy training/test
        x_lab = [x_lab for x_lab in range(len(Acc_train_vec))]
        plt.plot(x_lab, Acc_train_vec, color='r', label='training accuracy')
        plt.plot(x_lab, Acc_test_vec, color='b', label='test accuracy')
        plt.axvline(x=idx, color='g', linestyle='-')
        plt.title('Accuracy of Database _{}'.format(self.kfold.model.file))
        plt.legend(loc='upper right')
        plt.legend()
        plt.show()

        return W_arr[idx, :, 0], Acc_train_vec[idx], Acc_test_vec[idx]

    def remove_feature(self):
        '''
            Greedy algorithm to find the combination of features with the highest accuracy
            Step1: Choose the best feature to train the model
            Step2: Add another feature and train the model
            Step 3: If accuracy increases -> keep that new feature, else remove new feature
            Repeat step 2-3 until all features have been tested.
        '''
        # load input file
        data_table = self.kfold.model.load_input()

        database = data_table.T

        row, col = np.shape(database)

        # training with only 1 feature 
        # greedy to choose the best one
        Y = database[row - 1, :]
        acc_train_kfold = []
        acc_test_kfold = []
        for i in range(row - 1):
            X = database[i, :].reshape(1, col)

            acc_train_previous, acc_test_previous = self.kfold.kfold_data_calculate(X, Y)
            acc_train_kfold.append(acc_train_previous)
            acc_test_kfold.append(acc_test_previous)

        # select feature with the best accuracy of test
        best_feature = np.argmax(acc_test_kfold)
        print('The best feature is feature: ', best_feature, " with an accuracy of: ", acc_test_kfold)
        idx_feature = [best_feature + 1]
        X = database[best_feature, :].reshape(1, col)

        for i in range(row - 1):
            if i == best_feature:
                continue

            new_feature = database[i, :].reshape(1, col)
            X_current = np.concatenate((X, new_feature), axis=0)
            acc_train_current, acc_test_current = self.kfold.kfold_data_calculate(X_current, Y)
            if (acc_test_current >= acc_test_previous) | (acc_train_current >= acc_train_previous):
                acc_test_previous = acc_test_current
                acc_train_previous = acc_train_current
                X = X_current
                idx_feature.append(i + 1)
            else:
                pass

        # Save remaining feature to .csv file
        new_data = np.concatenate((X, Y.reshape(1, col)), axis=0).T
        np.savetxt('{}/remove_new_{}'.format(self.kfold.model.path, self.kfold.model.file), new_data, delimiter=',')

        return acc_train_previous, acc_test_previous, idx_feature

    # =================================================================================
    def add_feature(self):
        '''
            Adds new features created by non-linear functions of existed features to try and improve accuracy
        '''
        # load input file
        data_table = self.kfold.model.load_input()

        database = data_table.T

        row, col = np.shape(database)

        # training with only 1 feature
        X = database[0:row - 1, :]
        Y = database[row - 1, :]

        acc_train_prev = 0
        acc_test_prev = 0
        idx_feature = []
        acc_train_vec = []
        acc_test_vec = []

        X_original = X

        # check for adding new features xi*xj
        for i in range(0, row - 1):
            new_feature = X_original[i - 1, :].reshape(1, col) * X_original[i, :].reshape(1, col)
            X_current = np.concatenate((X, new_feature), axis=0)
            acc_train_cur, acc_test_cur = self.kfold.kfold_data_calculate(X_current, Y)

            # if the new feature has a better accuracy, update X
            if (acc_test_cur >= acc_test_prev) | (acc_train_cur >= acc_train_prev):
                acc_test_prev = acc_test_cur
                acc_train_prev = acc_train_cur
                X = X_current
                idx_feature.append([i - 1, i])
                acc_train_vec.append(acc_train_prev)
                acc_test_vec.append(acc_test_prev)
            else:
                pass

        # check for adding new features xi**2    
        for i in range(0, row - 1):

            new_feature = np.square(X_original[i - 1, :].reshape(1, col))
            X_current = np.concatenate((X, new_feature), axis=0)
            acc_train_cur, acc_test_cur = self.kfold.kfold_data_calculate(X_current, Y)

            # if the new feature has a better accuracy, update X
            if (acc_test_cur >= acc_test_prev + 0.01) | (acc_train_cur >= acc_train_prev + 0.01):
                acc_test_prev = acc_test_cur
                acc_train_prev = acc_train_cur
                X = X_current
                idx_feature.append([i - 1, i - 1])
                acc_train_vec.append(acc_train_prev)
                acc_test_vec.append(acc_test_prev)
            else:
                pass

            # Save remaining feature to .csv file
            new_data = np.concatenate((X, Y.reshape(1, col)), axis=0).T
            np.savetxt('{}/add_new_{}'.format(self.kfold.model.path, self.kfold.model.file), new_data, delimiter=',')

        return acc_train_vec, acc_test_vec, idx_feature

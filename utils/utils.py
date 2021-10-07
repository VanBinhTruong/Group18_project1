
import numpy as np

def split_train_test(X, Y, rate):
    m, n = np.shape(X)  # m is number of feature, n is number of observation

    # make permutation for database
    seed = 0
    np.random.seed(seed)
    idx_per = np.random.permutation(n)
    Y_per = Y[idx_per].reshape(1, n)
    X_per = X[:, idx_per]

    # separate training set and validation set
    # take from index 0 to 80% of input X, Y as Training, otherwise for Validation
    temp = int(rate * n)
    X_train = X_per[:, 0:temp + 1]
    X_test = X_per[:, temp + 1:]

    Y_train = Y_per[:, 0:temp + 1]
    Y_test = Y_per[:, temp + 1:]

    return X_train, X_test, Y_train, Y_test

def accu_eval(Y_predict, Y_actual):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(Y_predict.T)):
        if Y_predict[0, i] == 1 and Y_actual[0, i] == 1:
            TP += 1
        elif Y_predict[0, i] == 0 and Y_actual[0, i] == 0:
            TN += 1
        elif Y_predict[0, i] == 0 and Y_actual[0, i] == 1:
            FN += 1
        elif Y_predict[0, i] == 1 and Y_actual[0, i] == 0:
            FP += 1
        else:
            raise ValueError('predict() error')

    accuracy = (TP + TN) * 1.0 / (TP + TN + FP + FN)

    return accuracy
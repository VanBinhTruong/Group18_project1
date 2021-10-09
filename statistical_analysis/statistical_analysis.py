import numpy as np
import matplotlib.pyplot as plt

class Statistical_Analysis():

    def init(self, DL_info = {}):
        '''
            Initiate value for parameters
            path: is folder of .csv files
        '''
        self.path = DL_info.get('path', 'database')
        self.file = DL_info.get('file', 'red_wine.csv')


    def load_input(self):
        '''
            path_csv is path of input.csv file
            EX: red_wine.csv, diabetes.csv
                diabet = np.genfromtxt('diabetes.csv', delimiter=",")
        '''
        path_csv = self.path + '/' + self.file
        training_data = np.genfromtxt(path_csv, delimiter=",")
        
        return training_data

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
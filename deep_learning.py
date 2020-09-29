# -*- coding: utf-8 -*-
"""
Using spyder because it has better code completion
"""

#%% Imports
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import preprocessing, metrics
import random

#Some balancing types
from imblearn.under_sampling import *
from imblearn.ensemble import *

#Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#%% Data loading and preprocessing
df = pd.read_csv('../df_clean.csv')
df = df.dropna()
df = df.drop(['yod', 'yoa', 'b_wt'], axis=1)

datatypes = {'age':'int8', 'sex':'category', 'ethnic':'category', 'pt_state':'category', 'raceethn':'category', 
             'campus':'category', 'admtype':'category', 'payer':'category', 'pay_ub92':'category', 
             'provider':'category', 'asource':'category', 'moa':'int8', 'service':'category', 'diag_adm':'category', 
             'los':'int8', 'los_binary':'category'}

df = df.astype(dtype=datatypes)

df['age'] = preprocessing.scale(df['age'])
df['moa'] = preprocessing.scale(df['moa'])

#%% Data splitting and balancing
x = df.drop(['los', 'los_binary'], axis=1)
x = pd.get_dummies(x)
y = df['los_binary']

def get_split(x, y, test_size=0.05, balancer=RandomUnderSampler):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    try:
        model = balancer()
    except:
        model = RandomUnderSampler()
    x_train, y_train = model.fit_resample(x_train, y_train)
    return (x_train, x_test, y_train, y_test);

def get_tensors(x, y):
    x = torch.tensor(x.values)
    y = torch.tensor(y.values)
    return x ,y;

#%% Other functions
def get_results(y_test, y_pred, results=None, model_name=None):
    [tn, fp, fn, tp] = metrics.confusion_matrix(y_test, y_pred).ravel()
    model_results = {'Model':model_name,
                     'Sensitivity':[tp/(tp+fn)],
                     'Specificity':[tn/(tn+fp)],
                     'Balanced Accuracy':[.5*(tp/(tp+fn) + tn/(tn+fp))],
                     'Accuracy':[(tp+tn)/(tp+tn+fp+fn)]}
    if not results:
        results = pd.DataFrame(columns=list(model_results.keys))
    if not model_name:
        model_name = 'Model ' & str(len(results))
    results.append(model_results, ignore_index=True)
    return results

#%% Building Neural Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1845, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return F.log_softmax(x, dim=1)

class Loader():
    def __init__(self, x, y, batch_size=64, shuffle=True):
        if len(x) != len(y):
            raise ValueError('x and y must have same number of entries')
        if batch_size > len (x):
            raise ValueError('batch size cannot be larger than total entries')
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        self.indices = list(range(len(self.x)))
        return self
    
    def __next__(self):
        if len(self.indices) > 0:
            if self.shuffle:
                index_gen = lambda: int(random.random()*len(self.indices))
            else:
                index_gen = lambda: 0
                
            xs = []
            ys = []
            try:
                for i in range(min(self.batch_size, len(self.indices))):
                    index = self.indices.pop(index_gen())
                    xs.append(list(self.x.iloc[index]))
                    ys.append(self.y[index])
                t = (torch.tensor(xs), torch.tensor(ys))
                return t
            except:
                print(f'{i},\n{index},\n{xs},\n{ys}')
        
        else:
            raise StopAsyncIteration


#%% Prepare data for training
data = get_split(x, y)

#%% Train test split
trainset = Loader(data[0], data[2])
testset = Loader(data[1], data[3])


#%% Training network
net = Net()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        pass







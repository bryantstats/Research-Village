# -*- coding: utf-8 -*-
"""
Using spyder because it has better code completion
"""

#%% Imports
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
import random

#Preprocessing pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#Some balancing types
from imblearn.under_sampling import *
from imblearn.ensemble import *

#Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2

#%% Data loading and preprocessing
INCLUDE_b_wt = True

numeric_features = ['age', 'los', 'b_wt', 'moa']
numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

categorical_features = ['sex', 'ethnic', 'pt_state', 'raceethn', 'campus', 'admtype', 'payer', 'pay_ub92', 
                        'provider', 'asource', 'service', 'diag_adm', 'los_binary']
categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

if INCLUDE_b_wt:
    df = pd.read_csv('../df_clean_v2_b_wt_fixed.csv')
    df = df.dropna()
    df = df.drop(['yod', 'yoa'], axis=1)
    
    datatypes = {'age':'int8', 'sex':'category', 'ethnic':'category', 'pt_state':'category', 'raceethn':'category',
                 'campus':'category', 'admtype':'category', 'payer':'category', 'pay_ub92':'category',
                 'provider':'category', 'asource':'category', 'moa':'int8', 'service':'category', 'diag_adm':'category',
                 'los':'int8', 'los_binary':'category', 'b_wt':'int16'}
    
    df = df.astype(dtype=datatypes)
    
    df['age'] = preprocessing.scale(df['age'])
    df['moa'] = preprocessing.scale(df['moa'])
    df['b_wt'] = preprocessing.scale(df['b_wt'])
    
else:
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

#%% Data splitting, balancing, and result calculating functions
def get_split(x, y, test_size=0.05, balancer=RandomUnderSampler):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    try:
        sampler = balancer()
    except:
        sampler = RandomUnderSampler()
    x_train, y_train = sampler.fit_resample(x_train, y_train)
    return (x_train, x_test, y_train, y_test)

def get_tensors(x, y):
    x = torch.tensor(x.values)
    y = torch.tensor(y.values)
    return x ,y;

def get_results(y_test, y_pred, results=None, model_name=None):
    [tn, fp, fn, tp] = metrics.confusion_matrix(y_test, y_pred).ravel()
    
    if not model_name:
        try:
            model_name = 'Model ' & str(len(results))
        except:
            model_name = 'Model 0'
    
    model_results = {'Model':model_name,
                     'Sensitivity':tp/(tp+fn),
                     'Specificity':tn/(tn+fp),
                     'Balanced Accuracy':.5*(tp/(tp+fn) + tn/(tn+fp)),
                     'Accuracy':(tp+tn)/(tp+tn+fp+fn)}
    if not results:
        results = pd.DataFrame(columns=list(model_results.keys()))
    results = results.append(model_results, ignore_index=True)
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

class HospitalDataset(Dataset):
    def __init__(self, x, y):
        if len(x) != len(y):
            raise ValueError("X and y must be same dimensions")
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.tensor(self.x.iloc[index])
        y = torch.tensor(self.y.iloc[index])
        return x, y

    def __len__(self):
        return len(self.x)

"""
Don't use this class! It was made before I understood how to subclass Dataset
and use the built-in DataLoader class
"""
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
            for i in range(min(self.batch_size, len(self.indices))):
                index = self.indices.pop(index_gen())
                xs.append(list(self.x.iloc[index]))
                ys.append(self.y.iloc[index])
            t = (torch.tensor(xs), torch.tensor(ys))
            return t

        else:
            raise StopIteration

    def change_batch(self, new_size):
        self.batch_size = new_size

#%% Prepare data for training
x_data = df.drop(['los', 'los_binary'], axis=1)
x_data = pd.get_dummies(x_data)
y_data = df['los_binary']

balanced_data = get_split(x_data, y_data)

#%% Train test split
# trainset = Loader(balanced_data[0], balanced_data[2])
# testset = Loader(balanced_data[1], balanced_data[3])
train_data = HospitalDataset(balanced_data[0], balanced_data[2])
test_data = HospitalDataset(balanced_data[1], balanced_data[3])

BATCH_SIZE = 128
trainset = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
testset = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

#%% Training network
net = Net()

#List of learning rate and epochs to be run
training_periods = {0:[1e-3, 3]}

batch_losses = []
epoch_losses = []

for p in training_periods:
    learning_rate = training_periods[p][0]
    epochs = training_periods[p][1]
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for data in trainset:
            net.zero_grad()
            x, y = data
            output = net(x.view(-1, 1845).float())
            loss = F.cross_entropy(output, y)
            batch_losses.append(loss)
            loss.backward()
            optimizer.step()
        print(loss)
        epoch_losses.append(loss)


#Previous segmented training sessions
# =============================================================================
# "Initial training"
# optimizer = optim.Adam(net.parameters(), lr=8e-3)
# for epoch in range(EPOCHS):
#     for data in trainset:
#         x, y = data
#         net.zero_grad()
#         output = net(x.view(-1, 1845))
#         loss = F.nll_loss(output, y)
#         batch_losses.append(loss)
#         loss.backward()
#         optimizer.step()
#     epoch_losses.append(loss)
# 
# "Secondary training"
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
# for epoch in range(EPOCHS):
#     for data in trainset:
#         x, y = data
#         net.zero_grad()
#         output = net(x.view(-1, 1845))
#         loss = F.nll_loss(output, y)
#         batch_losses[1].append(loss)
#         loss.backward()
#         optimizer.step()
#     epoch_losses[1].append(loss)
# =============================================================================

#%% Testing accuracy
correct = 0
total = 0

y_pred = []

with torch.no_grad():
    for data in testset:
        x, y = data
        output = net(x.view(-1, 1845).float())
        for pred in output:
            y_pred.append(int(torch.argmax(pred)))
        # for idx, i in enumerate(output):
        #     if torch.argmax(i) == y[idx]:
        #         correct +=1
        #     total +=1

# print(f'Accuracy: {correct/total}')
res = get_results(list(balanced_data[3]), y_pred)

#%% Plotting loss
step = 8
plt.plot(batch_losses[::step] + batch_losses[::step])
plt.title(f'Every other {step}th batch loss')
plt.show()

#%% Moving average of loss
ma_loss = []
window_size = 64
for i in range(len(batch_losses) - window_size):
    ma_loss.append(sum(batch_losses[i:i+window_size])/window_size)

plt.plot(ma_loss)
plt.title(f'Moving Average of Loss - Window={window_size}')


#%% Testing

d = {'5': 110096,
             '8': 105616,
             '9': 221948,
             '1': 154721,
             '0': 365763,
             '7': 104125,
             '6': 104850,
             '2': 104656,
             '3': 106024,
             '4': 162591}

o = {}
for i in sorted(d):
    o[i] = d[i]

asd = sorted(d)
dsa = [d[i] for i in asd]
plt.bar(asd, dsa)
plt.show()

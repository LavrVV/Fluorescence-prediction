import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import sys
import os
import pydicom
import pickle

from PIL import Image

from torch import optim
import torch
import torch.cuda
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import models
import gc

N = 200

attrs = [(0x8, 0x103e),
(0x8, 0x1140),
(0x18, 0x20),
(0x18, 0x21),
(0x18, 0x22),
(0x18, 0x23),
(0x18, 0x1030),
(0x18, 0x1312),
(0x28, 0x106),
(0x28, 0x107),
(0x43, 0x102f),
(0x8, 0x1030),
(0x8, 0x103e),
(0x10, 0x10)#patirnt name
        ]

def extract_data(datafiles, sh=128, d3=True):
    data = []
    y = []
    other = []
    flourescented = ['DEMKIN A.P.', 'ZHURAVLEVA E.V.', 'MOVLATOVA L.B.', 'PETROV S.V.', 'POPOVA I.I.', 'SAFRONOVA T.A.', 'ANTROPOVA E.P.', 
 'BRITIKOVA E.V.', 'GACHKAEVA S.V.', 'SHURUPOV V.YA.', 'KHAKIMOVA M.M.', 'DASHKINA R.F.']
    count = 0
    for fileDCM in datafiles:
        ds = pydicom.read_file(fileDCM)

        sex = ds[(0x10, 0x40)].value
        sex = 1 if sex == 'M' else 0
        age = ds[(0x10, 0x1010)].value
        age = int(age[:-1])
        weight = ds[(0x10, 0x1030)].value
        weight = float(weight)

        patient = ds[attrs[13]].value
        ds = ds.pixel_array
        if len(ds.shape) > 2:
            count += 1
            continue

        other.append([sex, age, weight])
        
        if ds.shape[0] != sh:
                
            gg = Image.fromarray(ds)
            gg = gg.resize((sh, sh))
            curr = np.asarray(gg)
            if not d3:
                curr = curr.reshape(sh**2)
        else:
            curr = ds
            if not d3:
                curr = curr.reshape(sh**2)
        mmax = float(np.max(curr))
        if mmax != 0:
            curr = curr / mmax 
        if d3:
            data.append([curr])
        else:
            data.append(curr)
        if patient in flourescented:
            y.append(0)
        else:
            y.append(1)
    print('3d im count', count)
    data = np.array(data, dtype=float)
    other = np.array(other)
    return data, y, other 

def train(network, data, epochs, learning_rate, l2=0):
    loss = nn.MSELoss()
    #SGD
    network.cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=l2)
    train, test = train_test_split(data)
    train_loader = DataLoader(train, batch_size=50, shuffle=True)
    test_loader = DataLoader(test, batch_size=50, shuffle=True)
    res = 0
    try:
        for epoch in range(epochs):
            train_losses = []
            train_accuracies = []
            
            for X in train_loader:
                X = Variable(X)
                X = X.float()
                X = X.cuda()
                network.zero_grad()
                prediction = network(X)
                loss_batch = loss(prediction, X)
                loss_batch.backward()
                optimizer.step()
                train_losses.append(loss_batch.item())
                
                prediction = prediction.cpu()
                X = X.cpu()
                
                train_accuracies.append(np.absolute((prediction.data.numpy() - X.data.numpy())).sum()) 
            train_losses = np.mean(train_losses)
            train_accuracies = np.mean(train_accuracies)
            
            
            test_losses = []
            test_accuracies = []
            for X in test_loader:
                X = Variable(X)
                X = X.float()
                X = X.cuda()
                prediction = network(X)
                loss_batch = loss(prediction, X)
                test_losses.append(loss_batch.item())
                
                prediction = prediction.cpu()
                X = X.cpu()
                
                test_accuracies.append(np.absolute((prediction.data.numpy() - X.data.numpy())).sum())
            
            test_losses = np.mean(test_losses)
            test_accuracies = np.mean(test_accuracies)
            
            res = (train_losses, test_losses, train_accuracies, test_accuracies)
            sys.stdout.write('\rEpoch {0}... (Train/Test) MSE: {1:.3f}/{2:.3f}\tAccuracy: {3:.3f}/{4:.3f}'.format(
                        epoch, train_losses, test_losses,
                        train_accuracies, test_accuracies))
        
    except KeyboardInterrupt:
        pass
    network.cpu()
    return res

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
#from xgboost import XGBClassifier

def check_models(X, y, metric='accuracy'):
    res = {'LogisticRegression': 0,
           'DecisionTree': 0,
           'GaussianNB': 0,
           'SVC': 0,
           'SGD': 0,
           'GaussianProcess': 0,
           'XGBoost': 0
          }
    #print metric
    X, y = shuffle(X, y)
    lg = LogisticRegression(solver='liblinear', multi_class='auto')
    cv = cross_validate(lg, X, y, cv=10, scoring=metric)
    res['LogisticRegression'] = np.mean(cv['test_score'])
    
    dt = DecisionTreeClassifier(criterion='gini')
    cv = cross_validate(dt, X, y, cv=10)
    res['DecisionTree'] = np.mean(cv['test_score'])
    
    bayes = GaussianNB()
    cv = cross_validate(bayes, X, y, cv=10, scoring=metric)
    res['GaussianNB'] = np.mean(cv['test_score'])
    
    svc = SVC(kernel='rbf', shrinking=True, probability=False, degree=3, gamma='auto')
    cv = cross_validate(svc, X, y, cv=10, scoring=metric)
    res['SVC'] = np.mean(cv['test_score'])
    
    sgd = SGDClassifier(tol=1e-3, max_iter=1000, loss='hinge')
    cv = cross_validate(sgd, X, y, cv=10, scoring=metric)
    res['SGD'] = np.mean(cv['test_score'])
    
    gp = GaussianProcessClassifier()
    cv = cross_validate(gp, X, y, cv=10, scoring=metric)
    res['GaussianProcess'] = np.mean(cv['test_score'])
    
    #gb = XGBClassifier(max_depth=10, n_estimators=20, booster='dart')
    #cv = cross_validate(gb, X, y, cv=10, scoring=metric)
    #res['XGBoost'] = np.mean(cv['test_score'])
    
    return res

if __name__ == "__main__":
    datafiles = []
    f = open('file_list.txt', 'r', encoding='utf-8')
    for i in f:
        datafiles.append(i[:-1])
    f.close()

    
    
    with open('y.pickle', 'rb') as f:
	    y = pickle.load(f)
    with open('other.pickle', 'rb') as f:
	    other = pickle.load(f)
    
    #d2data64, y, other = extract_data(datafiles, sh=64, d3=False)
    #d2data128, y, other = extract_data(datafiles, sh=128, d3=False)
    #d3data128, y, other = extract_data(datafiles, sh=64, d3=True)
    #other = pd.DataFrame(other)
    
    print(len(y))
    print(other.shape)
    
    print('cuda')
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
    torch.cuda.device(2)
    print(torch.cuda.current_device())
    torch.cuda.set_device(2)
    print(torch.cuda.current_device())
    
    res = open('out.txt', 'w')
    #TODO
    myd2models64 = {}
    myd2models128 = {}
    myd3models128 = {}

    d2models64 = [models.MyAutoencoder64, models.MySparceAutoencoder64, models.MyDeepAutoencoder64, models.MyDeepSparceAutoencoder64]
    d2models128 = [models.MyAutoencoder128, models.MySparceAutoencoder128, models.MyDeepAutoencoder128, models.MyDeepSparceAutoencoder128]


    #with open('d2data64.pickle', 'rb') as f:
    #    d2data64 = pickle.load(f)
    #print(d2data64.shape)
#
    #for i in range(10, 21):
    #    for model in d2models64:
    #        currmodel = model(i)
    #        train_loss, test_loss, train_ac, test_ac = train(currmodel, d2data64, N, 0.001)
    #        res.write('*' * 20 + '\n')
    #        res.write(str(type(currmodel)) + str(i) + '\n')
    #        res.write('(Train/Test) MSE: {0:.3f}/{1:.3f}\tAccuracy: {2:.3f}/{3:.3f}\n'.format(
    #                    train_loss, test_loss,
    #                    train_ac, test_ac))
    #    
    #        X1 = currmodel.encode(Variable(torch.Tensor(d2data64)))
    #        X1 = pd.DataFrame(X1)
    #        data = pd.concat([X1, other], axis=1)
    #        f1 = check_models(data, y, metric='f1')
#
    #        res.write(str(f1) + '\n')
#
    #del d2data64
    #gc.collect()


    with open('d2data128.pickle', 'rb') as f:
        d2data128 = pickle.load(f)
    
    for i in range(10, 21):
        for model in d2models128:
            currmodel = model(i)
            train_loss, test_loss, train_ac, test_ac = train(currmodel, d2data128, N, 0.001)
            res.write('*' * 20 + '\n')
            res.write(str(type(currmodel)) + str(i) + '\n')
            res.write('(Train/Test) MSE: {0:.3f}/{1:.3f}\tAccuracy: {2:.3f}/{3:.3f}\n'.format(
                        train_loss, test_loss,
                        train_ac, test_ac))
        
            X1 = currmodel.encode(Variable(torch.Tensor(d2data128)))
            X1 = pd.DataFrame(X1)
            data = pd.concat([X1, other], axis=1)
            f1 = check_models(data, y, metric='f1')

            res.write(str(f1) + '\n')

    del d2data128
    gc.collect()

    with open('d3data128.pickle', 'rb') as f:
        d3data128 = pickle.load(f)
    
    for i in range(10, 21):
        currmodel = models.MyCAutoencoder(i)
        train_loss, test_loss, train_ac, test_ac = train(currmodel, d3data128, N, 0.001)
        res.write('*' * 20 + '\n')
        res.write(str(type(currmodel)) + str(i) + '\n')
        res.write('(Train/Test) MSE: {0:.3f}/{1:.3f}\tAccuracy: {2:.3f}/{3:.3f}\n'.format(
                    train_loss, test_loss,
                    train_ac, test_ac))
    
        X1 = currmodel.encode(Variable(torch.Tensor(d3data128)))
        X1 = pd.DataFrame(X1)
        data = pd.concat([X1, other], axis=1)
        f1 = check_models(data, y, metric='f1')
        res.write(str(f1) + '\n')

    res.close()


#    for i in range(10, 21):
#        myd2models64['MyAutoencoder64(' + str(i) + ')'] = models.MyAutoencoder64(i)
#        myd2models64['MySparceAutoencoder64(' + str(i) + ')'] = models.MySparceAutoencoder64(i)    
#        myd2models64['MyDeepAutoencoder64(' + str(i) + ')'] = models.MyDeepAutoencoder64(i)
#        myd2models64['MyDeepSparceAutoencoder64(' + str(i) + ')'] = models.MyDeepSparceAutoencoder64(i)
#
#        myd2models128['MyAutoencoder128(' + str(i) + ')'] = models.MyAutoencoder128(i)
#        myd2models128['MySparceAutoencoder128(' + str(i) + ')'] = models.MySparceAutoencoder128(i)    
#        myd2models128['MyDeepAutoencoder128(' + str(i) + ')'] = models.MyDeepAutoencoder128(i)
#        myd2models128['MyDeepSparceAutoencoder128(' + str(i) + ')'] = models.MyDeepSparceAutoencoder128(i)
#
#        myd3models128['MyCAutoencoder(' + str(i) + ')'] = models.MyCAutoencoder(i)
#
#
#    for model_name, model in myd2models64.items():
#        print(model_name)
#        train_loss, test_loss, train_ac, test_ac = train(model, d2data64, N, 0.001)
#        res.write('*' * 20)
#        res.write(model_name)
#        res.write('(Train/Test) MSE: {0:.3f}/{1:.3f}\tAccuracy: {2:.3f}/{3:.3f}'.format(
#                        train_loss, test_loss,
#                        train_ac, test_ac))
#        
#        X1 = model.encode(Variable(torch.Tensor(d2data64)))
#        data = pd.concat([X1, other], axis=1)
#        f1 = check_models(data, y, metric='f1')
#
#        res.write(f1)
#
#    for model_name, model in myd2models128.items():
#        train_loss, test_loss, train_ac, test_ac = train(model, d2data128, N, 0.001)
#        res.write('*' * 20)
#        res.write(model_name)
#        res.write('(Train/Test) MSE: {0:.3f}/{1:.3f}\tAccuracy: {2:.3f}/{3:.3f}'.format(
#                        train_loss, test_loss,
#                        train_ac, test_ac))
#        
#        X1 = model.encode(Variable(torch.Tensor(d2data128)))
#        data = pd.concat([X1, other], axis=1)
#        f1 = check_models(data, y, metric='f1')
#
#        res.write(f1)
#
#    for model_name, model in myd3models128.items():
#        train_loss, test_loss, train_ac, test_ac = train(model, d3data128, N, 0.001)
#        res.write('*' * 20)
#        res.write(model_name)
#        res.write('(Train/Test) MSE: {0:.3f}/{1:.3f}\tAccuracy: {2:.3f}/{3:.3f}'.format(
#                        train_loss, test_loss,
#                        train_ac, test_ac))
#        
#        X1 = model.encode(Variable(torch.Tensor(d3data128)))
#        data = pd.concat([X1, other], axis=1)
#        f1 = check_models(data, y, metric='f1')
#
#        res.write(f1)
#
#    res.close()


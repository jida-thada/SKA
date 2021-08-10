import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import csv
import matplotlib.pyplot as plt 
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
                   'axes.labelsize': 'x-large',
                   'axes.titlesize':'x-large',
                   'xtick.labelsize':'x-large',
                   'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)



def plot_acc(plt_name,train_acc_ep,test_acc_ep,save_path='./'):
    plt.figure(figsize=(15,10))
    plt.subplot(223)
    plt.ylim([0, 1.05])
    plt.plot(train_acc_ep, 'b-', label='training')
    plt.plot(test_acc_ep, 'r--', label='testing')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt_name = plt_name+".png"
    plt.savefig(save_path+plt_name, bbox_inches='tight')
    plt.clf()

def plot_loss(plt_name,losses,save_path='./'):
    plt.figure(figsize=(15,10))
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(save_path+plt_name, bbox_inches='tight')
    plt.clf()

def write_row(row, name, path="./", round=False):
    if round:
        row = [np.round(i, 2) for i in row]
    f = path + name + ".csv"
    with open(f, "a+") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",")
        filewriter.writerow(row)

def stratified_sampling(n_splits,test_size,x,y,seed=0):
    try:
        shuffle_data = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed) 
        for train_index, test_index in shuffle_data.split(x, y): 
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
        return x_train, x_test, y_train, y_test
    except: 
        test_size += 0.01
        shuffle_data = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed) 
        for train_index, test_index in shuffle_data.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
        return x_train, x_test, y_train, y_test

def read_data(filepath):
    x = []; y = []
    with open(filepath, 'r') as f:
        for line in f:
            l = [float(x) for x in line.split()]
            y.append(int(l[0]))
            x.append(l[1:])
    return x, y

def labelmap(classlist):
    j = 0 
    labels_map = {}
    for i in classlist:
        labels_map[i] = j 
        j += 1
    return labels_map

def prepare_y_TTL(num_teacher,y_with_labels,teachers_class):
    target_size = [len(y_with_labels),num_teacher]
    yttl = torch.zeros(target_size[0],target_size[1])
    for i in range(len(y_with_labels)):
        real_class = y_with_labels[i]
        for j in range(num_teacher):
            if real_class in teachers_class[j]:
                yttl[i,j] = 1
    return yttl

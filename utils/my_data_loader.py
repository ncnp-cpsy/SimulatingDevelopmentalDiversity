import torch
import numpy as np
import pandas as pd
from utils.my_utils import *

class MySoftmaxTransformer(object):
    def __init__(self):
        self.ref = 10
        self.sigma = 0.05
        self.lim = [-1, 1]
        self.interval = (self.lim[1] - self.lim[0]) / (self.ref - 1)
        self.reference = [self.lim[0] + self.interval * i for i in range(self.ref)]

    def __call__(self, sample):
        temp = np.exp( (- (np.array(self.reference) - np.array(sample)) ** 2) / self.sigma)
        rslt = temp / np.sum(temp)
        return rslt.tolist()

class MySoftmaxTransformer_2(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        rslt = softmax_transform(data=sample, mode='forward')

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_num, transform=None, filename='data', sep=',', max_time_step=None):
        self.transform = transform
        self.data_num = data_num
        self.data = []
        self.label = []

        for i in range(self.data_num):
            filepass = filename + str(i)
            # data = np.loadtxt(filepass,
            #                   delimiter='\t', # if use space-splited, change this
            #                   usecols=range(1,11))
            data = np.loadtxt(filepass,
                              delimiter=sep)
            if max_time_step==None: data = torch.from_numpy(data[:,:])
            else: data = torch.from_numpy(data[:max_time_step,:])
            data = data.to(torch.float)
            
            self.data.append(data)
            self.label.append(i) # For learning of initial value, sequence number is given to RNN

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label


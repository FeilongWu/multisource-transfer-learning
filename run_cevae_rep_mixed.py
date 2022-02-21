'''
This program runs CEVAE and calculates the statistics.
Author: WU Feilong
'''

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch
from sklearn.model_selection import train_test_split
import csv
import numpy as np
from argparse import ArgumentParser
from CEVAE_pt_synthetic_mixed import *
import matplotlib.pyplot as plt
import copy
import random




def getargs():
    parser = ArgumentParser()
    parser.add_argument('-lr', type=float, default=0.002)
    parser.add_argument('-opt', choices=['adam', 'rmsprop'], default='adam')
    parser.add_argument('-epochs', type=int, default=60)
    parser.add_argument('-atc', type=str, default='relu')
    parser.add_argument('-nf', type=int, default=20) # num of features
    parser.add_argument('-d', type=int, default=20) # dim of z
    parser.add_argument('-bs', type=int, default=40) # batch size
    
    return parser.parse_args()

def read_data(path):
    # data in csv

    data = []
    all_y = []
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            temp = [int(row[0])] # treatment
            all_y.append(float(row[1])) # y factual
            for i in row[1:]:
                temp.append(float(i))
            data.append(temp)
    return data

def normal_data(data):
    # normalize y to standard normal
    # return normalized data, y mean and y std
    all_y = []
    for i in data:
        all_y.append(i[1])
    y_mu = np.mean(all_y)
    y_std = np.std(all_y)
    for i in data:
        i[1] = (i[1] - y_mu) / y_std
    return data, y_mu, y_std

class createDS(Dataset):
    def __init__(self, data, device, mixed=False):
        self.data = data
        self.device = device
        self.mixed = mixed
        if mixed:
            self.source = len(data)
    def __len__(self):
        if self.mixed:
            return len(self.data[0])
        else:
            return len(self.data)
    def __getitem__(self, idx):
        if self.mixed:
            dic = {}
            for i in range(self.source):
                idx1 = str(i)
                row = self.data[i][idx]
                dic[idx1 + 'T'] = row[0]
                dic[idx1 + 'y_fac'] = row[1]
                dic[idx1 + 'mu0'] = row[3]
                dic[idx1 + 'mu1'] = row[4]
                dic[idx1 + 'x'] = torch.tensor(row[5:])
            return dic
        else:
            row = self.data[idx]
            dic = {'T':row[0]}
            dic['y_fac'] = row[1]
            #dic['y_cf'] = torch.tensor([row[2]])
            dic['mu0'] = row[3]
            dic['mu1'] = row[4]
            dic['x'] = torch.tensor(row[5:])
            return dic


def get_scores(moedel, data, stat, source):
    ITE_pre = []
    ITE_gt = []
    ite_f_cf = [] # record ite as difference between pre and ground truth
    source_s = {}
    for i in range(source):
        source_s[i] = {'ITE_pre': [], 'ITE_gt': [], 'ite_f_cf': []}
    for _, batch in enumerate(data):
        for i in range(source):
            #mu = stat[i][0]
            #std = stat[i][1]
            idx = str(i)
            x = batch[idx + 'x'].to(device)
            T = batch[idx + 'T'].numpy()
            y_fc = batch[idx + 'y_fac'].numpy()
            y0_gt = batch[idx + 'mu0'].numpy()
            y1_gt = batch[idx + 'mu1'].numpy()
            y0_pre, y1_pre = model.predict(x, i) # i: select specific nn
            y0_pre = y0_pre.cpu().detach().numpy() 
            y1_pre = y1_pre.cpu().detach().numpy() 
            ITE_gt1 = y1_gt - y0_gt
            ITE_gt.extend(ITE_gt1)
            source_s[i]['ITE_gt'].extend(ITE_gt1)
            ITE_pre1 = y1_pre - y0_pre
            ITE_pre.extend(ITE_pre1)
            source_s[i]['ITE_pre'].extend(ITE_pre1)
            ite_f_cf1 = (y1_pre - y_fc) * (1 - T) + (y_fc  - y0_pre) * T
            ite_f_cf.extend(ite_f_cf1)
            source_s[i]['ite_f_cf'].extend(ite_f_cf1)
    ITE_pre = np.array(ITE_pre)
    ITE_gt = np.array(ITE_gt)
    ite_f_cf = np.array(ite_f_cf)
        
    ITE = np.sqrt(np.mean(np.square(ITE_gt - ite_f_cf))) # rmse ite
    ATE = np.abs(np.mean(ITE_pre) - np.mean(ITE_gt)) # abs ate
    PEHE = np.sqrt(np.mean(np.square(ITE_gt - ITE_pre)))

    socres_s1 = {}
    for i in source_s:
        socres_s1[i] = {}
        socres_s1[i]['ITE'] = np.sqrt(np.mean(np.square(np.array(source_s[i]['ITE_gt']) - \
                                                     np.array(source_s[i]['ite_f_cf']))))
        socres_s1[i]['ATE'] = np.abs(np.mean(np.array(source_s[i]['ITE_pre'])) \
                                     - np.mean(np.array(source_s[i]['ITE_gt'])))
        socres_s1[i]['PEHE'] = np.sqrt(np.mean(np.square(np.array(source_s[i]['ITE_gt']) - \
                                                         np.array(source_s[i]['ITE_pre']))))
    return ITE, ATE, PEHE, socres_s1

def runmodel(model, data_tr, data_val, data_te, epochs, opt, lr, save_p,  \
            stat, source):
    if opt.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt.lower() == 'rmsprop':
        optimizer = torch.optim.RMSProp(model.parameters(), lr=lr)
    else:
        print('Unknown optimizer!')
        exit(0)
    PEHE_best = np.inf
    ATE_best = np.inf
    ITE_best = np.inf
    loss0 = torch.tensor(float('inf')).to(device)
    model.train()
    x_axis = []
    y_axis = []
    for epoch in range(epochs):
        x_axis.append(epoch)
        temp_loss = []
        loss1 = torch.tensor(0.).to(device)
        for _, batch in enumerate(data_tr):
            for i in range(source):
                idx = str(i)
                T = batch[idx + 'T'].to(device)
                y1 = batch[idx + 'y_fac']
                y = []
                for entry in y1:
                    y.append([float(entry)])
                y = torch.tensor(y).to(device)
                x = batch[idx + 'x'].to(device)
                loss = torch.mean(model.getloss(x, y, T, i)) # i: select specific nn
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss1 += loss.detach()
        #y_axis.append(sum(temp_loss) / len(temp_loss))
        # evaluation
        if loss1 > loss0:
            continue
        else:
            loss0 = loss1
        model.eval()
        ITE, ATE, PEHE, _ = get_scores(model, data_val, stat, source)
        
        
        if PEHE_best >= PEHE and ATE_best >= ATE and ITE_best >= ITE:
            print('model saved at epoch: ' + str(epoch))
            print('Val ITE:' + str(ITE) + ', Val ATE: ' + str(ATE) + \
                  ', Val PEHE: ' + str(PEHE))
            PEHE_best = PEHE
            ATE_best = ATE
            ITE_best = ITE

            # test
            ITE_te, ATE_te, PEHE_te, source_s = get_scores(model, data_te, stat, source)
            
           # torch.save(model.state_dict(), save_p)
    print('Test ITE:' + str(ITE_te) + ', Test ATE: ' + str(ATE_te) + \
          ', Test PEHE: ' + str(PEHE_te))
    return ITE_te, ATE_te, PEHE_te, source_s
##    plt.plot(x_axis, y_axis)
##    plt.xlabel('Epoch')
##    plt.ylabel('Loss')
##    plt.show()

def print_result(scores, source_s):
    for i in source_s:
        print('Results in source ' + str(i) + ': ')
        ITE = np.array(source_s[i]['ITE'])
        ATE = np.array(source_s[i]['ATE'])
        PEHE = np.array(source_s[i]['PEHE'])
        print('ITE: ' + str(np.mean(ITE)) + ' +- ' + str(np.std(ITE)) + ', ATE: '\
              + str(np.mean(ATE)) + ' +- ' + str(np.std(ATE)) + ', PEHE; ' + \
              str(np.mean(PEHE)) + ' +- ' + str(np.std(PEHE)))
    ITE, ATE, PEHE = [], [], []
    for i in scores:
        ITE.append(scores[i]['ITE'])
        ATE.append(scores[i]['ATE'])
        PEHE.append(scores[i]['PEHE'])
    mu_ITE, std_ITE = np.mean(ITE), np.std(ITE)
    mu_ATE, std_ATE = np.mean(ATE), np.std(ATE)
    mu_PEHE, std_PEHE = np.mean(PEHE), np.std(PEHE)
    print('Results on all replications. ITE: ' + str(mu_ITE) + ' +- ' + \
          str(std_ITE) + ', ATE: ' + str(mu_ATE) + ' +- ' + str(std_ATE) + \
          ', PEHE: ' + str(mu_PEHE) + ' +- ' + str(std_PEHE))
    

if __name__ == "__main__":
    # control randomness
    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(3)
    ######
    
    #train_ratio = 0.8
    test_ratio = 0.2
    save_path = './model1.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = getargs()
    replication = 2
    source = 1
    scores = {}
    tr_size = 400
    source_s = {}
    trDL, valDL, teDL, stat =[], [], [], {}
    for i in range(source):
        source_s[i] = {'ITE': [], 'ATE': [], 'PEHE': []}
        path = './source' + str(i) + '.csv'
        data = read_data(path)
        data_tr, data_te = train_test_split(data, test_size=test_ratio)
        data_val, data_te = train_test_split(data_te, test_size=0.5)
        #data_tr, y_mu, y_std = normal_data(copy.deepcopy(data_tr[:tr_size]))
        #stat[i] = [y_mu, y_std]
        trDL.append(data_tr[:tr_size])
        valDL.append(data_val)
        teDL.append(data_te)
    trainDS = createDS(trDL, device, mixed=True)
    validDS = createDS(valDL, device, mixed=True)
    testDS = createDS(teDL, device, mixed=True)
    trainDL = DataLoader(trainDS, batch_size = args.bs, shuffle=True)
    validDL = DataLoader(validDS, batch_size=50)
    testDL = DataLoader(testDS, batch_size=50)
    
    for i in range(replication):
        print('replication: ' + str(i + 1))
        model = CEVAE(args.nf, args.d, device, source)
        ITE, ATE, PEHE, source_s1 = runmodel(model, trainDL, validDL, testDL, \
                                  args.epochs, args.opt, args.lr,save_path,
                                  stat, source)
        scores[i] = {}
        scores[i]['ITE'] = ITE
        scores[i]['ATE'] = ATE
        scores[i]['PEHE'] = PEHE
        for k in source_s1:
            source_s[k]['ITE'].append(source_s1[k]['ITE'])
            source_s[k]['ATE'].append(source_s1[k]['ATE'])
            source_s[k]['PEHE'].append(source_s1[k]['PEHE'])
    print_result(scores, source_s)

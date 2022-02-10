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
from CEVAE_idph import *
import matplotlib.pyplot as plt
import copy
import random




def getargs():
    parser = ArgumentParser()
    parser.add_argument('-lr', type=float, default=0.002)
    parser.add_argument('-opt', choices=['adam', 'rmsprop'], default='adam')
    parser.add_argument('-epochs', type=int, default=60)
    parser.add_argument('-atc', type=str, default='relu')
    parser.add_argument('-nfcon', type=int, default=6) # num of continuous features
    parser.add_argument('-nfbin', type=int, default=19) # num of binary features
    parser.add_argument('-d', type=int, default=20) # dim of z
    parser.add_argument('-bs', type=int, default=50) # batch size
    
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
    def __init__(self, data, device):
        self.data = data
        self.device = device
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data[idx]
        dic = {'T':row[0]}
        dic['y_fac'] = row[1]
        #dic['y_cf'] = torch.tensor([row[2]])
        dic['mu0'] = row[3]
        dic['mu1'] = row[4]
        dic['xcon'] = torch.tensor(row[5:11])
        dic['xbin'] = torch.tensor(row[11:])
        return dic

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def get_scores(moedel, data, y_mu, y_std):
    ITE_pre = []
    ITE_gt = []
    ite_f_cf = [] # record ite as difference between pre and ground truth
    
    for _, batch in enumerate(data):
##        bs = len(batch['T'])
##        for i in range(bs):
##            x = batch['x'][i].to(device)
##            T = batch['T'][i]
##            y_fc = batch['y_fac'][i]
##            y0_gt = batch['mu0'][i]
##            y1_gt = batch['mu1'][i]
##            y0_pre, y1_pre = model.predict(x)
##            y0_pre = y0_pre * y_std + y_mu
##            y1_pre = y1_pre * y_std + y_mu
##            ITE_gt.append(float(y1_gt) - float(y0_gt))
##            ITE_pre.append(float(y1_pre) - float(y0_pre))
##            if T == 0:
##                ite_f_cf.append(float(y1_pre) - float(y_fc))
##            else:
##                ite_f_cf.append(float(y_fc) - float(y0_pre))
        xcon = batch['xcon'].to(device)
        xbin = batch['xbin'].to(device)
        T = batch['T'].numpy()
        y_fc = batch['y_fac'].numpy()
        y0_gt = batch['mu0'].numpy()
        y1_gt = batch['mu1'].numpy()
        y0_pre, y1_pre = model.predict(xcon, xbin)
        y0_pre = y0_pre.cpu().detach().numpy() * y_std + y_mu
        y1_pre = y1_pre.cpu().detach().numpy() * y_std + y_mu
        ITE_gt.extend(y1_gt - y0_gt)
        ITE_pre.extend(y1_pre - y0_pre)
        ite_f_cf.extend((y1_pre - y_fc) * (1 - T) + (y_fc  - y0_pre) * T)
    ITE_pre = np.array(ITE_pre)
    ITE_gt = np.array(ITE_gt)
    ite_f_cf = np.array(ite_f_cf)
        
    ITE = np.sqrt(np.mean(np.square(ITE_gt - ite_f_cf))) # rmse ite
    ATE = np.abs(np.mean(ITE_pre) - np.mean(ITE_gt)) # abs ate
    PEHE = np.sqrt(np.mean(np.square(ITE_gt - ITE_pre)))
    return ITE, ATE, PEHE

def runmodel(model, data_tr, data_val, data_te, epochs, opt, lr, save_p,  \
            y_mu, y_std ):
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
            T = batch['T'].to(device)
            y1 = torch.tensor(batch['y_fac']).to(device)
            y = []
            for entry in y1:
                y.append([float(entry)])
            y = torch.tensor(y).to(device)
            xcon = batch['xcon'].to(device)
            xbin = batch['xbin'].to(device)
            loss = torch.mean(model.getloss(xcon, xbin, y, T))
            temp_loss.append(float(loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss1 += loss.detach()
        y_axis.append(sum(temp_loss) / len(temp_loss))
        # evaluation
        if loss1 > loss0:
            continue
        else:
            loss0 = loss1
        model.eval()
        ITE, ATE, PEHE = get_scores(model, data_val, y_mu, y_std)
        
        
        if PEHE_best >= PEHE and ATE_best >= ATE and ITE_best >= ITE:
            print('model saved at epoch: ' + str(epoch))
            print('Val ITE:' + str(ITE) + ', Val ATE: ' + str(ATE) + \
                  ', Val PEHE: ' + str(PEHE))
            PEHE_best = PEHE
            ATE_best = ATE
            ITE_best = ITE
            

            # test
            ITE_te, ATE_te, PEHE_te = get_scores(model, data_te, y_mu, y_std)
            
           # torch.save(model.state_dict(), save_p)
    print('Test ITE:' + str(ITE_te) + ', Test ATE: ' + str(ATE_te) + \
          ', Test PEHE: ' + str(PEHE_te))
##    plt.plot(x_axis, y_axis)
##    plt.xlabel('Epoch')
##    plt.ylabel('Loss')
##    plt.show()
    return ITE_te, ATE_te, PEHE_te
    

def print_result(scores, lrs):
    PEHE = {}
    for i,j in enumerate(lrs):
        PEHE[j] = []
        for k in scores:
            PEHE[j].append(scores[k]['PEHE'][i])
    mean_PEHE = []
    for i in PEHE:
        mean_PEHE.append((np.mean(PEHE[i]), i))
    mean_PEHE = sorted(mean_PEHE)
    lr = mean_PEHE[0][1]
    print('learning rate = ' + str(lr))
    idx = lrs.index(lr)
    ITE, ATE, PEHE = [], [], []
    for i in scores:
        ITE.append(scores[i]['ITE'][idx])
        ATE.append(scores[i]['ATE'][idx])
        PEHE.append(scores[i]['PEHE'][idx])
    mu_ITE, std_ITE = np.mean(ITE), np.std(ITE)
    mu_ATE, std_ATE = np.mean(ATE), np.std(ATE)
    mu_PEHE, std_PEHE = np.mean(PEHE), np.std(PEHE)
    print('Results on all replications. ITE: ' + str(mu_ITE) + ' +- ' + \
          str(std_ITE) + ', ATE: ' + str(mu_ATE) + ' +- ' + str(std_ATE) + \
          ', PEHE: ' + str(mu_PEHE) + ' +- ' + str(std_PEHE))
    
def save_plot(scores, s):
    mu, std = [], []
    ITE, ATE, PEHE = [], [], []
    for i in scores:
        ITE.append(scores[i]['ITE'])
        ATE.append(scores[i]['ATE'])
        PEHE.append(scores[i]['PEHE'])
    x = scores[i]['size']
    ITE = np.array(ITE)
    ATE = np.array(ATE)
    PEHE = np.array(PEHE)
    ITEmu,ITEstd, ATEmu, ATEstd, PEHEmu, PEHEstd = [], [], [], [], [], []
    col = len(ITE[0])
    for i in range(col):
        ITEmu.append(np.mean(ITE[:, i]))
        ITEstd.append(np.std(ITE[:, i]))
        ATEmu.append(np.mean(ATE[:, i]))
        ATEstd.append(np.std(ATE[:, i]))
        PEHEmu.append(np.mean(PEHE[:, i]))
        PEHEstd.append(np.std(PEHE[:, i]))
    ATEmu = np.array(ATEmu)
    ATEstd = np.array(ATEstd)
    PEHEmu = np.array(PEHEmu)
    PEHEstd = np.array(PEHEstd)
    #print((ATEmu, ATEstd))
    #print((PEHEmu, PEHEstd))
    plt.figure(s)
    plt.errorbar(x, ITEmu, yerr=ITEstd, label='ITE', color='red')
    
    plt.errorbar(x, ATEmu, yerr=ATEstd, label='ATE', color='blue')
    plt.plot(x, ATEmu+ATEstd, marker='v', ls='', color='blue')
    plt.plot(x, ATEmu-ATEstd, marker='v', ls='', color='blue')
    
    plt.errorbar(x, PEHEmu, yerr=PEHEstd, label='PEHE', color='limegreen')
    plt.plot(x, PEHEmu+PEHEstd, marker='v', ls='', color='limegreen')
    plt.plot(x, PEHEmu-PEHEstd, marker='v', ls='', color='limegreen')
    plt.xlabel('Training Size')
    plt.ylabel('Test Error')
    plt.legend()
    plt.savefig('./source' + str(s) + '.png')
    print('ITE mean: ' + str(ITEmu))
    print('ITE std: ' + str(ITEstd))
    print('ATE mean: ' + str(ATEmu))
    print('ATE std: ' + str(ATEstd))
    print('PEHE mean: ' + str(PEHEmu))
    print('PEHE std: ' + str(PEHEstd))

if __name__ == "__main__":
    # control randomness
    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(3)
    ######
    
    #train_ratio = 0.8
    test_ratio = 0.2
    path = './ihdp_npci_'
    save_path = './model1.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = getargs()
    #lrs = [0.001, 0.002, 0.004]
    replication = 3
    #ratios = [0.2/10, 0.4/10, 0.6/10, 0.8/10, 1/10]
    ratios = [400]
    for s in range(1,2):
        path = './ihdp_npci_' + str(s) + '.csv'
        data = read_data(path)
        scores = {}
        data_tr, data_te = train_test_split(data, test_size=test_ratio)
        data_val, data_te = train_test_split(data_te, test_size=0.5)
        validDS = createDS(data_val, device)
        testDS = createDS(data_te, device)
        validDL = DataLoader(validDS, batch_size=30)
        testDL = DataLoader(testDS, batch_size=30)
        for i in range(replication):
        #print('replication: ' + str(i + 1))
##            data_tr, data_te = train_test_split(data, test_size=test_ratio)
##            data_val, data_te = train_test_split(data_te, test_size=0.5)
##            validDS = createDS(data_val, device)
##
##            testDS = createDS(data_te, device)
##            validDL = DataLoader(validDS, batch_size=30)
##            testDL = DataLoader(testDS, batch_size=30)
            scores[i] = {'ITE': [], 'ATE': [], 'PEHE': [],'size': []}
            for ratio in ratios:
                #train_size = int(len(data_tr) * ratio)
                train_size = ratio
                data_tr1, y_mu, y_std = normal_data(copy.deepcopy(data_tr))
                trainDS = createDS(data_tr1, device)
                trainDL = DataLoader(trainDS, batch_size = args.bs, shuffle=True)
                model = CEVAE(args.nfcon, args.nfbin, args.d, device)
                #model.apply(init_weights)
                ITE, ATE, PEHE = runmodel(model, trainDL, validDL, testDL, \
                                      args.epochs, args.opt, args.lr ,save_path,\
                                          y_mu, y_std)
                scores[i]['ITE'].append(ITE)
                scores[i]['ATE'].append(ATE)
                scores[i]['PEHE'].append(PEHE)
                scores[i]['size'].append(train_size)
        save_plot(scores, s)
    

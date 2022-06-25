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
from CEVAE_pt_synthetic_bin import *
import copy
import random
import matplotlib.pyplot as plt




def getargs():
    parser = ArgumentParser()
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-opt', choices=['adam', 'rmsprop'], default='adam')
    parser.add_argument('-epochs', type=int, default=90)
    parser.add_argument('-atc', type=str, default='softplus')
    parser.add_argument('-nf', type=int, default=5) # nuclsm of features
    parser.add_argument('-d', type=int, default=20) # dim of z
    parser.add_argument('-bs', type=int, default=100) # batch size
    
    return parser.parse_args()

def read_data(path):
    # data in csv

    data = []
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            temp = [int(row[0])] # treatment
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
        dic['x'] = torch.tensor(row[5:])
        return dic

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def get_scores(model, data, y_mu, y_std, device):
    ITE_pre = []
    ITE_gt = []
    ite_f_cf = [] # record ite as difference between pre and ground truth
    
    for _, batch in enumerate(data):
        x = batch['x'].to(device)
        T = batch['T'].numpy()
        y_fc = batch['y_fac'].numpy()
        y0_gt = batch['mu0'].numpy()
        y1_gt = batch['mu1'].numpy()
        y0_pre, y1_pre = model.predict(x)
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

def early_stop(current, last, tol):
    if (current - last) >= tol:
        return True
    else:
        return False

def runmodel(model, data_tr, data_val, data_te, epochs, opt, lr, save_p,  \
            y_mu, y_std,device, ear_stop=30):
    if opt.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    elif opt.lower() == 'rmsprop':
        optimizer = torch.optim.RMSProp(model.parameters(), lr=lr, weight_decay=0.0001)
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
            y1 = torch.tensor(batch['y_fac']).to(device).float()
            y = y1.view(y1.size()[0], 1)
            x = batch['x'].to(device)
            loss = model.getloss(x, y, T)
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
        ITE, ATE, PEHE = get_scores(model, data_val, y_mu, y_std, device)
        
        
        if PEHE_best >= PEHE and ATE_best >= ATE:
      #      print('model saved at epoch: ' + str(epoch))
      #      print('Val ITE:' + str(ITE) + ', Val ATE: ' + str(ATE) + \
      #            ', Val PEHE: ' + str(PEHE))
            PEHE_best = PEHE
            ATE_best = ATE
            ITE_best = ITE
            

            # test
            ITE_te, ATE_te, PEHE_te = get_scores(model, data_te, y_mu, y_std, device)
            if str(save_size) in save_p:
                torch.save(model.state_dict(), save_p)
            last_update = epoch
        if early_stop(epoch, last_update, ear_stop):
            print('Early stopped')
            break
    #print('Test ITE:' + str(ITE_te) + ', Test ATE: ' + str(ATE_te) + \
    #      ', Test PEHE: ' + str(PEHE_te))

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
    # scores: {rep1:{metrics}, rep2:{metrics}, ...}
    mu, std = [], []
    ITE, ATE, PEHE = [], [], []
    for i in scores: # key is replication
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
    out = ''
    out += 'ITE mean: ' + str(ITEmu) + '\n'
    out += 'ITE std: ' + str(ITEstd) + '\n'
    out += 'ATE mean: ' + str(ATEmu) + '\n'
    out += 'ATE std: ' + str(ATEstd) + '\n'
    out += 'PEHE mean: ' + str(PEHEmu) + '\n'
    out += 'PEHE std: ' + str(PEHEstd) + '\n'
    return out, (ITEmu[0], ITEstd[0], ATEmu[0], ATEstd[0], PEHEmu[0], PEHEstd[0])

def write_result(path, metrics, exp):
    file = open(path, 'a')
    file.write('\n')
    file.write('++++++++++++++++++++++Experiment++++++++++++++++++++++\n')
    file.write(exp + '\n')
    file.write(str(metrics))
    file.write('\n\n')
    file.close()

def single_dataset(s, replication, ratio, file, save_p):
    path =  './source' + str(s) + '.csv'
    data = read_data(path)
    scores = {}
    for i in range(replication):
        #print('replication: ' + str(i + 1))
        data_tr, data_te = train_test_split(data, test_size=test_ratio)
        data_val, data_te = train_test_split(data_te, test_size=0.5)
        validDS = createDS(data_val, device)
        testDS = createDS(data_te, device)
        validDL = DataLoader(validDS, batch_size=30)
        testDL = DataLoader(testDS, batch_size=30)
        scores[i] = {'ITE': [], 'ATE': [], 'PEHE': [],'size': []}
                #train_size = int(len(data_tr) * ratio)
        train_size = ratio
        data_tr1, y_mu, y_std = copy.deepcopy(data_tr[:train_size]), 0, 1
        trainDS = createDS(data_tr1, device)
        trainDL = DataLoader(trainDS, batch_size = args.bs, shuffle=True)
        model = CEVAE(args.nf, args.d, device, dl=120).to(device)
                #model.apply(init_weights)
        
        ITE, ATE, PEHE = runmodel(model, trainDL, validDL, testDL, \
                                      args.epochs, args.opt, lrate ,save_p,\
                                          y_mu, y_std, device)
        scores[i]['ITE'].append(ITE)
        scores[i]['ATE'].append(ATE)
        scores[i]['PEHE'].append(PEHE)
        scores[i]['size'].append(train_size)
    s1, scores = save_plot(scores, s)
    exp = 'Dataset: ' + str(s) + ', ' + str('Size; ') + str(ratio) 
    write_result(file, s1, exp)
    return scores

def plot_all(scores):
    figure, axis = plt.subplots(3,3)
    figure.tight_layout()
    n = -1
    for i in scores:
        dataset = 'dataset' + str(i)
        n += 1
        x = scores[i]['size']
        for col, j in enumerate(['ITE', 'ATE', 'PEHE']):
            y = scores[i][j+'mu']
            yerr = scores[i][j+'std']
            axis[n, col].errorbar(x, y , yerr=yerr, )
            axis[n, col].set_title(dataset + ' ' + j)
    plt.savefig('./grad_plot.png')

if __name__ == '__main__':
    # control randomness
    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(3)
    ######

    file = open('./baseline.txt', 'w')
    file.write('')
    file.close()

    metrics = {'ITE_mu':[], 'ITE_std':[], 'ATE_mu':[], 'ATE_std':[], 'PEHE_mu':[], 'PEHE_std':[]}
    #train_ratio = 0.8
    test_ratio = 0.21
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = getargs()
    lrs = [0.005]
    replication = 5
    #ratios = [0.2/10, 0.4/10, 0.6/10, 0.8/10, 1/10]
    ratios = [20]
    ###
    save_size = 450 # the size under which the model will be saved
    ###
    source = 1
    #save_p = './model2.pth'
    plot_s = {}
    for i in range(1,2):
        plot_s[i] = {'size':[], 'ITEmu':[], 'ITEstd':[], 'ATEmu':[], 'ATEstd':[],\
                     'PEHEmu':[], 'PEHEstd':[]}
    for s in range(1,2):
        for lrate, ratio in zip(lrs, ratios):
            save_p = './baseline_dataset' + str(s) + '_size' + str(ratio) + '.pth'
            scores  = single_dataset(s, replication, ratio, './baseline.txt', \
                                     save_p)
            plot_s[s]['size'].append(ratio)
            plot_s[s]['ITEmu'].append(scores[0])
            plot_s[s]['ITEstd'].append(scores[1])
            plot_s[s]['ATEmu'].append(scores[2])
            plot_s[s]['ATEstd'].append(scores[3])
            plot_s[s]['PEHEmu'].append(scores[4])
            plot_s[s]['PEHEstd'].append(scores[5])
    # plot_s: {'source1':{'size':[10,], 'ITEmu':[0,], 'ATEmu':[0,],..}}
    plot_all(plot_s)
            

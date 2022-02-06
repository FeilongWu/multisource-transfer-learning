'''
Use Pytorch to implement the original CEVAE
Author: WU Feilong
'''

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

def gen_randm(x):
    n = x ** 2
    matrix = np.random.rand(x, x)
    while np.linalg.cond(matrix) > n:
        matrix = np.random.rand(x, x)
    return matrix

def neur_network(insize, outsize, h, dh, act, zero_one = False, bias=True, nonneg=None):
    # zero_one: output in [0, 1]
    if act.lower() == 'relu':
        actv = nn.ReLU()
    elif act.lower() == 'sigmoid':
        actv = nn.Sigmoid()
    net = [nn.Linear(insize, dh, bias=bias), actv]
    for i in range(h):
        net.append(nn.Linear(dh, dh, bias=bias))
        net.append(actv)
    net.append(nn.Linear(dh, outsize, bias=bias))
    if zero_one:
        if outsize > 1:
            net.append(nn.Softmax())
        else:
            net.append(nn.Sigmoid())
    elif nonneg:
        net.append(actv)
    return nn.Sequential(*net)

class Network(nn.Module):
    def __init__(self, insize, outsize, h, dh, act, dist, zero_one = False, bias=True, nonneg=None):
        super(Network, self).__init__()
        # dist = "Bernoulli" / "Normal"
        self.nn = neur_network(insize, outsize, h, dh, act, zero_one=zero_one, bias=bias, nonneg=nonneg)
        if dist.lower() == 'bernoulli':
            self.dist = torch.distributions.Bernoulli
        elif dist.lower() == 'multinormal' or dist.lower() == 'multigaussian':
            self.dist = torch.distributions.MultivariateNormal
        elif dist.lower() == 'normal' or dist.lower() == 'gaussian':
            self.dist = torch.distributions.normal.Normal
        else:
            print('Unknown distribution!')
            exit(0)
        
    #@staticmethod
    def find_dist(self, p1, p2 = None):
        # p1, p2 are params of dist
##        if self.dist == 'bernoulli':
##            dist = torch.distributions.Bernoulli
##        elif self.dist == 'normal' or self.dist == 'gaussian':
##            dist = torch.distributions.MultivariateNormal
##        else:
##            print('Unknown distribution!')
##            exit(0)
            
        if p2 is not None:
            distribution = self.dist(p1,p2)
            #distribution = dist(p1,p2)
        else:
            distribution = self.dist(p1)
            #distribution = dist(p1)
        
        return distribution
    
    def sample(self, p1, p2 = None):
        # p1, p2 are params of dist
        
        if p2 is not None:
            noise = torch.randn_like(p1)
            sample = p1 + noise * p2 # reparametrization
        else:
            distribution = self.find_dist(p1, p2)
            sample = distribution.sample()
        return sample
    
    def log_p(self, x, p1, p2 = None):
        distribution = self.find_dist(p1, p2)
        
        return distribution.log_prob(x)

    def forward(self, x):
        return self.nn(x)
    
class CEVAE(nn.Module):
    def __init__(self, nf, d, device, s, h=3, prior_mean = torch.zeros(2), \
                 prior_std = torch.eye(2), dh=200, act='ReLU'):
        # h: num of hidden layers
        # nf: dim of features
        # d: dim of z
        # h: # of hidden layers
        # dh: size of hidden layer
        
        super(CEVAE, self).__init__()
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.s = s # sources
        self.device = device
        self.randm = torch.tensor(gen_randm(2 * nf), requires_grad=False).to(device)
        if len(prior_mean) != d:
            self.prior_mean = torch.zeros(d)
            self.prior_std = torch.eye(d)
        # define P(z)
        self.prior = torch.distributions.MultivariateNormal(self.prior_mean.to(device),\
                                                            torch.abs(self.prior_std.to(device)))
                                                            
        # p(x|z)
        self.p_x_z = Network(d, 2 * nf, h, dh, act, 'multiNormal', nonneg=True, bias=False).to(device)

        # p(t|z)
        self.p_t_z = Network(d, 1, h, dh, act, 'Bernoulli', zero_one=True).to(device)

       # p(y|z,t=1)
        self.p_y_zt1 = Network(d, 1, h, dh, act, 'Normal').to(device)

        # p(y|z,t=0)
        self.p_y_zt0 = Network(d, 1, h, dh, act, 'Normal').to(device)

        # q(t|x)
        self.q_t_x = Network(nf, 1, h, dh, act, 'Bernoulli', zero_one=True).to(device)

        # shared representation of x
        self.rep_x = Network(nf, dh, 0, dh, act, 'Normal', zero_one=True).to(device)

         # q(y|x,t=1)
        self.q_y_xt1 = Network(dh, 1, h-1, dh, act, 'Normal').to(device)

        # q(y|x,t=0)
        self.q_y_xt0 = Network(dh, 1, h-1, dh, act, 'Normal').to(device)

        # shared representation of x and y
        self.rep_xy = Network(nf+1, dh, 0, dh, act, 'Normal', zero_one=True).to(device)

        # q(z|x,y,t=0)
        self.q_z_xyt0 = [Network(dh, 2 * d, h-1, dh, act, 'multiNormal').to(device) for i in range(s)]

        # q(z|x,y,t=1)

        self.q_z_xyt1 = [Network(dh, 2 * d, h-1, dh, act, 'multiNormal').to(device) for i in range(s)]

    def multi_diag(self, x):
        r = [torch.diag(x[0])]
        for i in x[1:]:
            r.append(torch.diag(i))
        return torch.stack(r)

    def getloss(self, x, y, t, idx):
        h_xy = self.rep_xy(torch.cat((x,y), 1))
        h_x = self.rep_x(x)
        z_mu, z_std = torch.chunk(torch.mul(self.q_z_xyt0[idx](h_xy).T, 1-t).T + \
                                  torch.mul(self.q_z_xyt1[idx](h_xy).T, t).T, 2, \
                                  dim=1)
        #z_std = self.multi_diag(z_std)
        sample_z = self.q_z_xyt0[0].sample(z_mu, torch.exp(0.5 * z_std)) # z_std = log var
        y_mu = torch.mul(self.p_y_zt0(sample_z).T, 1 - t).T +\
                                  torch.mul(self.p_y_zt1(sample_z).T, t).T
        # logp(y|z,t)
        logp_y_zt = self.p_y_zt0.log_p(y, y_mu, torch.ones(y_mu.shape).to(self.device)).flatten()

        z_std = self.multi_diag(torch.exp(z_std))
        # logq (z|x,y,t)
        logp_z_xyt = self.q_z_xyt0[0].log_p(sample_z, z_mu, z_std)

        qy_mu = torch.mul(self.q_y_xt0(h_x).T, 1-t).T + torch.mul(self.q_y_xt1(h_x).T, t).T
        # logq (y|x,t=0)
        logq_y_xt = self.q_y_xt0.log_p(y, qy_mu, torch.ones(qy_mu.shape).to(self.device)).flatten()
        
        pt = self.p_t_z(sample_z)
        # logp(t|z)
        log_t_z = torch.diagonal(self.p_t_z.log_p(t.float(), pt)) 

        x_mu, x_std = torch.chunk(self.p_x_z(sample_z), 2, dim=1)
        x_std = self.multi_diag(torch.exp(x_std))
        # logp(x|z)
        log_x_z = self.p_x_z.log_p(x, x_mu, x_std)

        # logp(z)
        logp_z = self.prior.log_prob(sample_z)

        qt = self.q_t_x(x)
        # logq (t|x)
        log_t_x = torch.diagonal(self.q_t_x.log_p(t.float(), qt))
        loss = - (log_x_z + log_t_z + logp_y_zt + logp_z - logp_z_xyt\
                 + log_t_x + logq_y_xt)
        return loss
    
    def predict(self, x, idx):
        t = torch.round(self.q_t_x(x)).T.flatten()
        rep_x = self.rep_x(x)
        qy = (torch.mul(self.q_y_xt0(rep_x).T, 1 - t) + \
             torch.mul(self.q_y_xt1(rep_x).T, t)).T
        rep_xy = self.rep_xy(torch.cat((x,qy), 1))
        qz, _ = torch.chunk(torch.mul(self.q_z_xyt0[idx](rep_xy).T, 1 - t).T + \
                            torch.mul(self.q_z_xyt1[idx](rep_xy).T, t).T, 2, dim=1)
        y = (torch.mul(self.p_y_zt0(qz).T, 1 - t) + \
            torch.mul(self.p_y_zt1(qz).T, t)).T
        t1 = 1 - t
        y_com = (torch.mul(self.p_y_zt0(qz).T, 1 - t1) + \
            torch.mul(self.p_y_zt1(qz).T, t1)).T
        y0 = (torch.mul(y.T, 1 - t) + torch.mul(y_com.T, 1 - t1)).T.flatten()
        y1 = (torch.mul(y.T, t) + torch.mul(y_com.T, t1)).T.flatten()
        return y0, y1

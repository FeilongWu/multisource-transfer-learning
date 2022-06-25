import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

eps = 1e-7

def find_act(act):
    # act: function name
    if act.lower() == 'relu':
        actv = nn.ReLU()
    elif act.lower() == 'sigmoid':
        actv = nn.Sigmoid()
    elif act.lower() == 'softplus':
        actv = nn.Softplus()
    return actv

class px(nn.Module):
    def __init__(self, dz, nf, h, dl, actv):
        super(px, self).__init__()
        net = [nn.Linear(dz, dl), actv]
        for i in range(h-1):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        self.NN = nn.Sequential(*net)
        self.loc = nn.Linear(dl, nf)
        self.scale = nn.Sequential(*[nn.Linear(dl, nf), nn.Softplus()]) 

    def forward(self, z):
        rep_x = self.NN(z)
        loc = self.loc(rep_x)
        scale = self.scale(rep_x)
        return loc, scale

class pt(nn.Module):
    def __init__(self, dz, out, h, dl, actv):
        super(pt, self).__init__()
        net = [nn.Linear(dz, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        net.append(nn.Linear(dl, out))
        net.append(nn.Sigmoid())
        self.NN = nn.Sequential(*net)

    def forward(self, z):
        loc_t = self.NN(z)
        return loc_t

class py0(nn.Module):
    def __init__(self, dz, out, h, dl, actv):
        super(py0, self).__init__()
        net = [nn.Linear(dz, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        net.append(nn.Linear(dl, out))
        net.append(nn.Sigmoid())
        self.NN = nn.Sequential(*net)

    def forward(self, z):
        loc_y = self.NN(z)
        return loc_y

class py1(nn.Module):
    def __init__(self, dz, out, h, dl, actv):
        super(py1, self).__init__()
        net = [nn.Linear(dz, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        net.append(nn.Linear(dl, out))
        net.append(nn.Sigmoid())
        self.NN = nn.Sequential(*net)

    def forward(self, z):
        loc_y = self.NN(z)
        return loc_y


class rep_xy(nn.Module):
    def __init__(self, dxy, out, h, dl, actv):
        super(rep_xy, self).__init__()
        net = [nn.Linear(dxy, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        self.NN = nn.Sequential(*net)

    def forward(self, xy):
        return self.NN(xy)

class qz0(nn.Module):
    def __init__(self, insize, dz, h, dl, actv):
        super(qz0, self).__init__()
        net = [nn.Linear(insize, dl), actv]
        for i in range(h-1):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        self.NN = nn.Sequential(*net)
        self.loc = nn.Linear(dl, dz)
        self.scale = nn.Sequential(*[nn.Linear(dl, dz), nn.Softplus()])

    def forward(self, rep_xy):
        rep = self.NN(rep_xy)
        loc = self.loc(rep) 
        scale = self.scale(rep) + eps
        return torch.cat((loc, scale), 1)

class qz1(nn.Module):
    def __init__(self, insize, dz, h, dl, actv):
        super(qz1, self).__init__()
        net = [nn.Linear(insize, dl), actv]
        for i in range(h-1):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        self.NN = nn.Sequential(*net)
        self.loc = nn.Linear(dl, dz) 
        self.scale = nn.Sequential(*[nn.Linear(dl, dz), nn.Softplus()])

    def forward(self, rep_xy):
        rep = self.NN(rep_xy)
        loc = self.loc(rep) 
        scale = self.scale(rep) + eps
        return torch.cat((loc, scale), 1)
        
class rep_x(nn.Module):
    def __init__(self, dx, out, h, dl, actv):
        super(rep_x, self).__init__()
        net = [nn.Linear(dx, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        self.NN = nn.Sequential(*net)

    def forward(self, x):
        return self.NN(x)

class qt(nn.Module):
    def __init__(self, nf, dt, h, dl, actv):
        super(qt, self).__init__()
        net = [nn.Linear(nf, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        net.append(nn.Linear(dl, dt))
        net.append(nn.Sigmoid())
        self.NN = nn.Sequential(*net)

    def forward(self, x):
        t_loc = self.NN(x)
        return t_loc

class qy0(nn.Module):
    def __init__(self, insize, dy, h, dl, actv):
        super(qy0, self).__init__()
        net = [nn.Linear(insize, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        net.append(nn.Linear(dl, dy))
        net.append(nn.Sigmoid())
        self.NN = nn.Sequential(*net)

    def forward(self, rep_x):
        y_loc = self.NN(rep_x)
        return y_loc

class qy1(nn.Module):
    def __init__(self, insize, dy, h, dl, actv):
        super(qy1, self).__init__()
        net = [nn.Linear(insize, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        net.append(nn.Linear(dl, dy))
        net.append(nn.Sigmoid())
        self.NN = nn.Sequential(*net)

    def forward(self, rep_x):
        y_loc = self.NN(rep_x)
        return y_loc

    
class CEVAE(nn.Module):
    def __init__(self, nf, dz, device, h=3, dl=100, act='softplus'):
        # h: num of hidden layers
        # nf: dim of features
        # dz: dim of z
        # h: # of hidden layers
        # dl: size of hidden layer

        super(CEVAE, self).__init__()
        self.device = device
        self.dz = dz
        actv = find_act(act)
        self.px = px(dz, nf, h, dl, actv)
        self.pt = pt(dz, 1, h-2, dl, actv)
        self.py0 = py0(dz, 1, h, dl, actv)
        self.py1 = py1(dz, 1, h, dl, actv)
        self.rep_xy = rep_xy(nf+1, dl, 1, dl, actv)
        self.rep_x = rep_x(nf, dl, 1, dl, actv)
        self.qz0 = qz0(dl, dz, h-1, dl, actv)
        self.qz1 = qz1(dl, dz, h-1, dl, actv)
        self.qt = qt(nf, 1, h-2, dl, actv)
        self.qy0 = qy0(dl, 1, h-1, dl, actv)
        self.qy1 = qy1(dl, 1, h-1, dl, actv)
        
        

    def getloss(self, x, y, t):
        dt = t.shape[0]
        self.pz = dist.Normal(torch.zeros(dt, self.dz).to(self.device),\
                         torch.ones(dt, self.dz).to(self.device))
        # encoding
        rep_xy = self.rep_xy(torch.cat((x,y), 1))
        qz_loc, qz_scale = torch.chunk(self.qz0(rep_xy) * (1-t.view(dt, 1)) + \
                                       self.qz1(rep_xy) * t.view(dt, 1), 2, dim=1)
        qz_dist = dist.Normal(qz_loc, qz_scale)
        qz_s = qz_dist.rsample()


        # reconstruct distributions
        px_loc, px_scale = self.px(qz_s)
        pt_loc = self.pt(qz_s)
        py_loc = self.y_mean(qz_s, t)

        # auxiliary distributions
        qt_loc = self.qt(x)
        qt_dist = dist.Bernoulli(qt_loc)
        rep_x = self.rep_x(x)
        qy_loc = self.qy0(rep_x) * (1-t.view(dt, 1)) + \
                 self.qy1(rep_x) * t.view(dt, 1)
        qy_dist = dist.Bernoulli(qy_loc)

        # loss

        logpx = torch.sum(dist.Normal(px_loc, px_scale).log_prob(x))
        logpt = torch.sum(dist.Bernoulli(pt_loc).log_prob(t.view(dt, 1).float()))
        logpy = torch.sum(dist.Bernoulli(py_loc).log_prob(y))

        logpz = self.pz.log_prob(qz_s)
        logqz = qz_dist.log_prob(qz_s)

        logqt = qt_dist.log_prob(t.float().view(dt, 1))
        logqy = qy_dist.log_prob(y)

        return (-logpx - logpt - logpy - torch.sum(logpz) + torch.sum(logqz)\
               - torch.sum(logqt) - torch.sum(logqy)) / dt


    def y_mean(self, z, t):
        dt = t.shape[0]
        return self.py0(z) * (1-t.view(dt,1)) + self.py1(z) * t.view(dt,1)

    def lower_bound(self, x, y, t):
        dt = t.shape[0]
        self.pz = dist.Normal(torch.zeros(dt, self.dz).to(self.device),\
                         torch.ones(dt, self.dz).to(self.device))
        # encoding

        qt_loc = self.qt(x)
        qt_dist = dist.Bernoulli(qt_loc)
        qt = qt_dist.sample()
        rep_x = self.rep_x(x)
        qy_loc = self.qy0(rep_x) * (1-qt.view(dt, 1)) + \
                 self.qy1(rep_x) * qt.view(dt, 1)
        qy_dist = dist.Bernoulli(qy_loc)
        qy = qy_dist.sample()
        
        rep_xy = self.rep_xy(torch.cat((x,qy), 1))
        qz_loc, qz_scale = torch.chunk(self.qz0(rep_xy) * (1-qt.view(dt, 1)) + \
                                       self.qz1(rep_xy) * qt.view(dt, 1), 2, dim=1)
        qz_dist = dist.Normal(qz_loc, qz_scale)
        qz_s = qz_dist.mean


        # reconstruct distributions
        px_loc, px_scale = self.px(qz_s)
        pt_loc = self.pt(qz_s)
        py_loc = self.y_mean(qz_s, t)
        

        # loss

        logpx = torch.sum(dist.Normal(px_loc, px_scale).log_prob(x))
        logpt = torch.sum(dist.Bernoulli(pt_loc).log_prob(t.view(dt, 1).float()))
        logpy = torch.sum(dist.Bernoulli(py_loc).log_prob(y))

        logpz = self.pz.log_prob(qz_s)
        logqz = qz_dist.log_prob(qz_s)


        return -(-logpx - logpt - logpy - torch.sum(logpz) + torch.sum(logqz))



    def predict(self, x, sample = 130):
##        qt_loc = self.qt(x)
##        qt_dist = dist.Bernoulli(qt_loc)
##        t = qt_dist.sample()
##        dt = t.shape[0]
##        rep_x = self.rep_x(x)
##        qy_loc = self.qy0(rep_x) * (1-t.view(dt, 1)) + \
##                 self.qy1(rep_x) * t.view(dt, 1)
##        qy_dist = dist.Bernoulli(qy_loc)
##        qy = qy_dist.sample()
##        rep_xy = self.rep_xy(torch.cat((x,qy), 1))
##        qz_loc, _ = torch.chunk(self.qz0(rep_xy) * (1-t.view(dt, 1)) + \
##                                self.qz1(rep_xy) * t.view(dt, 1), \
##                                2, dim=1)
##        y0 = dist.Bernoulli(self.py0(qz_loc).flatten()).sample()
##        y1 = dist.Bernoulli(self.py1(qz_loc).flatten()).sample()
##        return y0, y1


        colx = x.shape[1]
        qt_loc = self.qt(x)
        dt = qt_loc.shape[0]
        qt_dist = dist.Bernoulli(qt_loc)
        t = qt_dist.sample_n(sample).view(sample * dt, 1)
        rep_x = self.rep_x(x)
        coln = rep_x.shape[1]
        rep_x = rep_x.repeat(1, sample).view(sample * dt, coln)
        qy_loc = self.qy0(rep_x) * (1-t) + \
                 self.qy1(rep_x) * t
        qy_dist = dist.Bernoulli(qy_loc)
        qy = qy_dist.sample()
        rep_xy = self.rep_xy(torch.cat((x.repeat(1, sample).view(sample * dt, \
                                                                 colx),qy), 1))
        
        qz_loc, qz_scale = torch.chunk(self.qz0(rep_xy) * (1-t) + \
                                self.qz1(rep_xy) * t, \
                                2, dim=1)
        qz_dist = dist.Normal(qz_loc, qz_scale)
        qz_samples = qz_dist.rsample()

        mu0 = self.py0(qz_samples).view(dt, sample, 1)
        mu1 = self.py1(qz_samples).view(dt, sample, 1)
        y0 = torch.mean(mu0, dim=1).flatten()
        y1 = torch.mean(mu1, dim=1).flatten()
        return y0, y1
        

        



        

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
    elif act.lower() == 'elu':
        actv = nn.ELU()
    return actv

class px1(nn.Module):
    def __init__(self, dz, nf, h, dl, actv):
        super(px1, self).__init__()
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
        scale = self.scale(rep_x) + eps
        return loc, scale

class px2(nn.Module):
    def __init__(self, dz, nf, h, dl, actv):
        super(px2, self).__init__()
        net = [nn.Linear(dz, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        net.append(nn.Linear(dl, nf))
        net.append(nn.Sigmoid())
        self.NN = nn.Sequential(*net)
    def forward(self, z):
        loc_t = self.NN(z)
        return loc_t

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
        self.NN = nn.Sequential(*net)

    def forward(self, rep_x):
        y_loc = self.NN(rep_x)
        return y_loc

    
class CEVAE(nn.Module):
    def __init__(self, x_con, x_bin, dz, device, h=3, dl=100, act='elu'):
        # h: num of hidden layers
        # nf: dim of features
        # dz: dim of z
        # h: # of hidden layers
        # dl: size of hidden layer

        super(CEVAE, self).__init__()
        self.device = device
        self.dz = dz
        actv = find_act(act)
##        self.rep_z = rep_x(dz, dl, 1, dl, actv)
##        self.px1 = px1(dl, x_con, h-1, dl, actv)
##        self.px2 = px2(dl, x_bin, h-1, dl, actv)
##        self.pt = pt(dz, 1, h-1, dl, actv)
##        self.py0 = py0(dz, 1, h, dl, actv)
##        self.py1 = py1(dz, 1, h, dl, actv)
##        self.rep_xy = rep_xy(x_bin+x_con+1, dl, 1, dl, actv)
##        self.rep_x = rep_x(x_bin+x_con, dl, 1, dl, actv)
##        self.qz0 = qz0(dl, dz, h-1, dl, actv)
##        self.qz1 = qz1(dl, dz, h-1, dl, actv)
##        self.qt = qt(x_bin+x_con, 1, h-2, dl, actv)
##        self.qy0 = qy0(dl, 1, h-1, dl, actv)
##        self.qy1 = qy1(dl, 1, h-1, dl, actv)

        self.rep_z = rep_x(dz, dl, h-1, dl, actv)
        self.px1 = px1(dl, x_con, 1, dl, actv)
        self.px2 = px2(dl, x_bin, 1, dl, actv)
        self.pt = pt(dz, 1, 1, dl, actv)
        self.py0 = py0(dz, 1, h, dl, actv)
        self.py1 = py1(dz, 1, h, dl, actv)
        self.rep_xy = rep_xy(x_bin+x_con+1, dl, h-1, dl, actv)
        self.rep_x = rep_x(x_bin+x_con, dl, h-1, dl, actv)
        self.qz0 = qz0(dl, dz, 1, dl, actv)
        self.qz1 = qz1(dl, dz, 1, dl, actv)
        self.qt = qt(x_bin+x_con, 1, h-2, dl, actv)
        self.qy0 = qy0(dl, 1, 1, dl, actv)
        self.qy1 = qy1(dl, 1, 1, dl, actv)
        
        

    def getloss(self, x_con, x_bin, y, t):
        #print(('x_con', x_con.shape,x_con,'x_bin', x_bin.shape,x_bin, \
        #       'y',y.shape, y,'t',t.shape, t))
        dt = t.shape[0]
        self.pz = dist.Normal(torch.zeros(dt, self.dz).to(self.device),\
                         torch.ones(dt, self.dz).to(self.device))
        ones = torch.ones(dt).to(self.device).unsqueeze(-1)
        #print(('ones',ones.shape))
        # variational approximation
        x = torch.cat((x_con, x_bin), 1)
        #print(('x', x.shape, x))
        qt_loc = self.qt(x)
        #print(('qt_loc', qt_loc.shape, qt_loc))
        qt_dist = dist.Bernoulli(qt_loc)
        rep_x = self.rep_x(x)
        #print(('rep_x', rep_x.shape))
        qy_loc = self.qy0(rep_x) * (1-t) + \
                 self.qy1(rep_x) * t
        #print(('qy_loc', qy_loc.shape, qy_loc))
        qy_dist = dist.Normal(qy_loc, ones)

        rep_xy = self.rep_xy(torch.cat((x,y.float()), 1))
        #print(('rep_xy', rep_xy.shape))
        qz_loc, qz_scale = torch.chunk(self.qz0(rep_xy) * (1-t) + \
                                       self.qz1(rep_xy) * t, 2, dim=1)
        #print(('qz_loc', qz_loc.shape, qz_loc))
        qz_dist = dist.Normal(qz_loc, qz_scale)
        qz_s = qz_dist.rsample().float()
        #print(('qz_s', qz_s.shape, qz_s))

        # reconstruct distributions
        rep_z = self.rep_z(qz_s)
        #print(('rep_z', rep_z.shape))
        px1_loc, px1_scale = self.px1(rep_z)
        #print(('px1_loc', px1_loc.shape, px1_loc))
        px2_loc = self.px2(rep_z)
        #print(('px2_loc', px2_loc.shape, px2_loc))
        pt_loc = self.pt(qz_s)
        #print(('pt_loc', pt_loc.shape, pt_loc))
        py_loc = self.y_mean(qz_s, t)
        #print(('py_loc', py_loc.shape, py_loc))

        #loss
        
        logpx1 = torch.sum(dist.Normal(px1_loc, px1_scale).log_prob(x_con))
        logpx2 = torch.sum(dist.Bernoulli(px2_loc).log_prob(x_bin))
        logpt = torch.sum(dist.Bernoulli(pt_loc).log_prob(t))
        logpy = torch.sum(dist.Normal(py_loc, ones).log_prob(y))

        logpz = self.pz.log_prob(qz_s)
        logqz = qz_dist.log_prob(qz_s)

        logqt = qt_dist.log_prob(t.float().view(dt, 1))
        logqy = qy_dist.log_prob(y)
        #print(('logpx1', dist.Normal(px1_loc, px1_scale).log_prob(x_con).shape))
        #print(('logpx2', dist.Bernoulli(px2_loc).log_prob(x_bin).shape))
        #print(('logpt', dist.Bernoulli(pt_loc).log_prob(t).shape))
        #print(('logpy', dist.Normal(py_loc, ones).log_prob(y).shape))
        #print(('logpz', self.pz.log_prob(qz_s).shape))
        #print(('logqz', qz_dist.log_prob(qz_s).shape))
        #print(('logqt', qt_dist.log_prob(t).shape))
        #print(('logqy', qy_dist.log_prob(y).shape))



        return (-logpx1 - logpx2 - logpt - logpy - torch.sum(logpz) + torch.sum(logqz)\
               - torch.sum(logqt) - torch.sum(logqy)) / dt


    def y_mean(self, z, t):
        dt = t.shape[0]
        return self.py0(z) * (1-t.view(dt,1)) + self.py1(z) * t.view(dt,1)




    def predict(self, x_con, x_bin, sample = 100):
##        #print(('x_con', x_con))
##        #print(('x_bin', x_bin))
##        x = torch.cat((x_con,  x_bin), 1)
##        #print(('x', x))
##        qt_loc = self.qt(x)
##        #print(('qt_loc', qt_loc))
##        dt = qt_loc.shape[0]
##        qt_dist = dist.Bernoulli(qt_loc)
##        t = qt_dist.sample()
##        #print(('t', t))
##        rep_x = self.rep_x(x)
##        #print(('rep_x', rep_x.shape))
##        qy_loc = self.qy0(rep_x) * (1-t) + \
##                 self.qy1(rep_x) * t
##        qy_dist = dist.Normal(qy_loc, torch.ones(dt).to(self.device).unsqueeze(-1))
##        qy = qy_dist.sample()
##        #print(('qy', qy))
##        
##        rep_xy = self.rep_xy(torch.cat((x,qy), 1))
##        #print(('rep_xy', rep_xy.shape))
##        qz_loc, qz_scale = torch.chunk(self.qz0(rep_xy) * (1-t) + \
##                                self.qz1(rep_xy) * t, \
##                                2, dim=1)
##        qz_dist = dist.Normal(qz_loc, qz_scale)
##        #print(('qz_loc', qz_loc.shape))
##        qz_samples = qz_dist.sample(sample_shape = [sample])
##        #print(('qz_samples', qz_samples.shape))
##        #print(('y0', self.py0(qz_samples)))
##        y0 = torch.mean(self.py0(qz_samples), dim=0).flatten()
##        y1 = torch.mean(self.py1(qz_samples), dim=0).flatten()


        dt = x_con.shape[0]
        x = torch.cat((x_con,  x_bin), 1)
        y0, y1 = torch.zeros(dt, 1).to(self.device), torch.zeros(dt, 1).to(self.device)
        for i in range(sample):
            qt_loc = self.qt(x)
            qt_dist = dist.Bernoulli(qt_loc)
            t = qt_dist.sample()
            rep_x = self.rep_x(x)
            qy_loc = self.qy0(rep_x) * (1-t) + \
                 self.qy1(rep_x) * t
            qy_dist = dist.Normal(qy_loc, torch.ones(dt).to(self.device).unsqueeze(-1))
            qy = qy_dist.rsample()
            rep_xy = self.rep_xy(torch.cat((x,qy), 1))
            qz_loc, qz_scale = torch.chunk(self.qz0(rep_xy) * (1-t) + \
                                self.qz1(rep_xy) * t, \
                                2, dim=1)
            qz_dist = dist.Normal(qz_loc, qz_scale)
            qz_sample = qz_dist.sample()
            y0 += self.py0(qz_sample) / sample
            y1 += self.py1(qz_sample) / sample
        

        return y0.flatten(), y1.flatten()
        

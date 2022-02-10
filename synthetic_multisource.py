'''
Generate synthetic datasets with different latent variable distributions.

'''

import numpy as np
from scipy.stats import bernoulli
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import beta
from scipy.stats import gamma

import csv

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_data(dz, nf, n, x_cof, y_cof, z_mu):
    ones = np.ones(dz)
    result = []
    x_std = np.diag(np.zeros(nf) + 0.21)
    for i in range(n):
        z = bernoulli.rvs(z_mu)
        x_alpha = sigmoid(np.matmul(x_cof, z))
        x_beta = x_alpha * 0.5
        x = beta.rvs(x_alpha, x_beta)
        t = bernoulli.rvs(np.mean(0.75 * z + 0.25 * (1 - z)))
        mu0 = (np.matmul(y_cof, z) + (1 - 0) * np.mean(z) * 3).tolist()[0]
        mu1 = (np.matmul(y_cof, z) + (1 - 1) * np.mean(z) * 3).tolist()[0]
        if t == 0:
            y_factual = norm.rvs(mu0, mu0/10)
            y_cf = norm.rvs(mu1, mu1/10)
        else:
            y_factual = norm.rvs(mu1, mu1/10)
            y_cf = norm.rvs(mu0, mu0/10)
        row = [t, y_factual, y_cf, mu0, mu1]
        row.extend(x)
        result.append(row)
    return result

def export_csv(data, idx):
    path = './source' + str(idx) + '.csv'
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter = ',')
        for  i in data:
            writer.writerow(i)


if __name__ == '__main__':
    # z ~ bern
    # x ~ beta
    # t ~ bern
    # y ~ norm
    num_s = 3 # num of sources
    dz = 25 # dimension of z
    nf = 20 # dimension of x
    n = 3000 # num of samples
    x_cof = np.random.rand(nf, dz)
    y_cof = np.random.rand(1, dz)
    z_mean = [0.3, 0.6, 0.9]
    for i in range(num_s):
        z_mu = np.clip(z_mean[i] + np.random.rand(1, dz) / 20, 0 ,1)
        data = generate_data(dz, nf, n, x_cof, y_cof, z_mu)
        export_csv(data, i)

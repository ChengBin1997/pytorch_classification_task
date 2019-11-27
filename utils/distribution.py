# -*- coding: utf-8 -*-
# @Time    : 2019/3/25 15:27

# @Author  : ChengBin

# @Desc : ==============================================

# ======================================================

# @Project : Learn the distribution of label

# @FileName: distribution.py.py

# @Software: PyCharm
import torch
import math

class Gaussian(object):
    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sample(self):
        epsilon = self.normal.sample()
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

def getlabel(n_classes,epsilon):
    onehot = torch.eye(n_classes).cuda()
    dir_parm = (1 - epsilon) * onehot + epsilon * torch.ones_like(onehot).cuda() / n_classes
    label_gauss = torch.distributions.dirichlet.Dirichlet(dir_parm)
    return label_gauss.sample()




if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES']= '3'
    n_classes =10
    onehot = torch.eye(n_classes).cuda()

    epsilon = 0.2
    dir_parm=  (1 - epsilon) * onehot +  epsilon* torch.ones_like(onehot).cuda()  / n_classes
    print(dir_parm)
    label_gauss = torch.distributions.dirichlet.Dirichlet(dir_parm)
    x=[]
    y=[]

    print(label_gauss.sample())

    x.append(label_gauss.sample()[0])
    y.append(label_gauss.sample()[1])

    print(x,y)

    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    sns.jointplot(x=x,y=y,kind="kde")
    plt.show()
    '''
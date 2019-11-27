# -*- coding: utf-8 -*-
# @Time    : 2019/5/7 16:58

# @Author  : ChengBin

# @Desc : ==============================================

# ======================================================

# @Project : Label_embedding

# @FileName: MLP.py.py

# @Software: PyCharm

import torch.nn as nn
__all__ = ['mlp']
class MLP(nn.Module):
    def __init__(self,args):
        super(MLP, self).__init__()
        if args.dataset in ['MNIST']:
            chanel = 1
            class_num =10
        elif args.dataset in ['cifar10','svhn']:
            chanel = 3
            class_num = 10
        elif args.dataset in ['cifar100']:
            chanel = 3
            class_num = 100

        self.chanel = chanel
        self.fc1 = nn.Linear(1024*chanel, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, class_num)

    def forward(self, din):
        din = din.view(-1, 32 * 32 * self.chanel)
        dout = nn.functional.relu(self.fc1(din))
        dout = nn.functional.relu(self.fc2(dout))
        return self.fc3(dout)


def mlp(**kwargs):
    """
    Constructs a MLP model.
    """
    return MLP(**kwargs)
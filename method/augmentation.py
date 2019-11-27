# -*- coding: utf-8 -*-
# @Time    : 2019/9/20 16:07

# @Author  : ChengBin

# @Desc : ==============================================

# ======================================================

# @Project : Label_embedding

# @FileName: aaugmentation.py

# @Software: PyCharm

import torch
import numpy as np
from .base import *

class mix_up_criterion(Criterion):
    def __init__(self, alpha=1.0):
        super(mix_up_criterion, self).__init__()
        self.alpha = alpha
        self.lamda = 1
        self.y_rand = []


    def prepare(self, inputs, targets):
        x, y = inputs.cuda(), targets.cuda(async=True)
        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, self.y_rand = y, y[index]

        self.lamda = lam
        return mixed_x, y_a


    def __call__(self, preds, targets, idx, epoch):
        criterion = torch.nn.functional.cross_entropy
        y_a, y_b = targets, self.y_rand
        return self.lamda * criterion(preds, y_a) + (1 - self.lamda) * criterion(preds, y_b)



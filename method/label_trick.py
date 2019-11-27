# -*- coding: utf-8 -*-
# @Time    : 2019/9/20 19:59

# @Author  : ChengBin

# @Desc : ==============================================

# ======================================================

# @Project : Label_embedding

# @FileName: label_trick.py

# @Software: PyCharm

import torch
from .base import *

class label_smoothing_criterion(Criterion):
    def __init__(self,epsilon,num_class):
        super(label_smoothing_criterion,self).__init__()
        use_cuda=torch.cuda.is_available()
        one_hot = torch.eye(num_class)
        if use_cuda:
            one_hot = one_hot.cuda()
        self.smoothing_label = one_hot * (1 - epsilon) + torch.ones_like(one_hot) \
             * epsilon / num_class

    def __call__(self,preds,targets,idx,epoch):
        loss = CrossEntropyLoss(preds, self.smoothing_label[targets,:])
        return loss

    def process(self, epoch=None, args=None):
        print(self.smoothing_label)


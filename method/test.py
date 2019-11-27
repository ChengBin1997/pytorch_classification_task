# -*- coding: utf-8 -*-
# @Time    : 2019/9/20 21:16

# @Author  : ChengBin

# @Desc : ==============================================

# ======================================================

# @Project : Label_embedding

# @FileName: test.py.py

# @Software: PyCharm

import torch
import numpy as np
from .base import *

### mixup图片置信度降低
class correct_mix_up_criterion(Criterion):
    def __init__(self, alpha=1.0,pow=1,is_joint=True):
        super(correct_mix_up_criterion, self).__init__()
        self.alpha = alpha
        self.lamda = 1
        self.y_rand = []
        self.is_joint = is_joint
        self.pow = pow

    def prepare(self, inputs, targets):
        x, y = inputs.cuda(), targets.cuda(async=True)
        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()

        if self.is_joint:
            size1=round(lam*(x.size(2)-1)+0.5)
            a = torch.nn.functional.interpolate(x,size=[size1,x.size(3)])
            b = torch.nn.functional.interpolate(x[index, :], size=[x.size(2)-size1, x.size(3)])
            mixed_x = torch.cat([a,b],dim = 2)
        else:
            mixed_x = lam * x + (1 - lam) * x[index, :]

        y_a, self.y_rand = y, y[index]

        self.lamda = lam
        return mixed_x, y_a

    def __call__(self, preds, targets, idx, epoch):
        criterion = torch.nn.functional.cross_entropy
        y_a, y_b = targets, self.y_rand
        return (self.lamda**self.pow)*criterion(preds, y_a) + ((1-self.lamda)**self.pow) * criterion(preds, y_b)




## KL散度取反
class anti_kld_Criterion(Criterion):
    def __init__(self, epsilon, num_class):
        super(anti_kld_Criterion, self).__init__()
        use_cuda = torch.cuda.is_available()
        one_hot = torch.eye(num_class)
        if use_cuda:
            one_hot = one_hot.cuda()
        self.soft_label = one_hot * (1 - epsilon) + torch.ones_like(one_hot) \
                          * epsilon / num_class

    def __call__(self, preds, targets, idx, epoch):
        loss = KL_Divergence(torch.nn.functional.softmax(preds, 1) + 1e-5, self.soft_label[targets, :] + 1e-5)
        return loss


class VarianceControlProcessor():
    def __init__(self, num_samples, num_classes, delay_epoch=1):
        super(VarianceControlProcessor, self).__init__()
        self.delay_epoch = delay_epoch
        self.record_label = torch.zeros(size=(num_samples, num_classes)).type(torch.cuda.FloatTensor)


    def append(self, outputs, sample_idx):
        ans = F.softmax(outputs, dim=1)
        self.record_label[sample_idx, :] = ans

    def get_label(self, sample_idx):
        return self.record_label[sample_idx, :]

class VarianceControlCriterion(Criterion):
    def __init__(self, num_samples, num_classes, start_epoch, beta=1,is_count=False,stop_epoch=300):
        super(VarianceControlCriterion, self).__init__()
        self.beta = beta
        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch
        self.lprocessor = VarianceControlProcessor(num_samples, num_classes)
        self.loss = CrossEntropyLoss
        self.is_count = is_count
        print('is_count:',is_count)
        if not is_count:
            print('beta:',beta)
        print('Compute the score')

    def __call__(self, preds, targets, idx, epoch):
        class_mask = preds.new_zeros(size=preds.size()).cuda()
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)
        beta = self.beta

        with torch.no_grad():
            self.lprocessor.append(preds, idx)

        if epoch >= self.start_epoch:
            forepred = self.lprocessor.get_label(idx)
            if self.is_count:
                #beta =1- (preds*forepred+1e-8)/(preds**2+(preds-forepred)**2+1e-8)
                with torch.no_grad():
                    beta =  (preds * forepred+1e-8) / ((preds - class_mask) ** 2+1e-8)
                    #print(beta[0])
                    #input()
            preds = preds - beta * (forepred - class_mask)

        loss = CrossEntropyLoss(preds, class_mask)
        return loss

    def process(self, epoch=None, args=None):
        print('\n',self.lprocessor.get_label(0).data)
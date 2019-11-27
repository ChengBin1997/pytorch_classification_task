# -*- coding: utf-8 -*-
# @Time    : 2019/10/10 10:36

# @Author  : ChengBin

# @Desc : ==============================================

# ======================================================

# @Project : Label_embedding

# @FileName: distillation.py

# @Software: PyCharm

from .base import *

class ditill_criterion(Criterion):
    def __init__(self,tmodel,lambda_st,T,type='base',cls_w=1,num_classes=10):
        super(ditill_criterion,self).__init__()
        self.tmodel = tmodel
        self.lambda_st = lambda_st
        self.T = T
        self.cls_w = cls_w
        self.soft_label = []
        self.img_size =[]
        self.criterionCls = torch.nn.CrossEntropyLoss().cuda()
        self.criterionST = torch.nn.KLDivLoss(reduction='sum').cuda()
        self.type = type
        self.num_classes = num_classes
        if self.type == 'noise':
            self.T=1
            self.cls_w = 0
        elif self.type == 'mixup':
            self.mix_label =[]


    def __call__(self,preds,targets,idx,epoch):
        logits = torch.nn.functional.log_softmax(preds / self.T, dim=1)
        st_loss = self.criterionST(logits, self.soft_label)*(self.T *self.T )/self.img_size
        if self.type =='mixup':
            cls_loss = CrossEntropyLoss(preds, self.mix_label)
        else:
            cls_loss = self.criterionCls(preds, targets)

        loss = self.cls_w*cls_loss+self.lambda_st*st_loss

        return loss

    def class2one_hot(self,labels):
        class_mask = torch.zeros(labels.size(0),self.num_classes).cuda()
        label_ids = labels.view(-1, 1).cuda()
        class_mask.scatter_(1, label_ids, 1.)
        return class_mask

    def prepare(self, inputs, targets):
        if self.type=='base':
            x, y = inputs.cuda(), targets.cuda(async=True)
        elif self.type=='noise':
            x = torch.rand(size=inputs.size()).cuda()
            y = targets.cuda(async=True)
        elif self.type == 'mixup':
            x, y = inputs.cuda(), targets.cuda(async=True)
            one_hot_y = self.class2one_hot(targets)
            alpha = 1
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).cuda()

            x = lam * x + (1 - lam) * x[index, :]
            self.mix_label = lam * one_hot_y + (1 - lam) * one_hot_y[index, :]
        elif self.type == 'disturb':
            x = inputs + torch.rand(size=inputs.size())
            x, y = x.cuda(), targets.cuda(async=True)

        self.img_size = x.size(0)
        self.tmodel.eval()
        self.soft_label = torch.nn.functional.softmax(self.tmodel(x) / self.T, dim=1)

        return x,y


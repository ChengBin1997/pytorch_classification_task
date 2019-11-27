# -*- coding: utf-8 -*-
# @Time    : 2019/3/21 10:51

# @Author  : ChengBin

# @Desc : ==============================================

# ======================================================

# @Project : Learn the distribution of label

# @FileName: LR_adjust.py

# @Software: PyCharm

import numpy as np

class ramps:
    @staticmethod
    def sigmoid_rampup(current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    @staticmethod
    def linear_rampup(current, rampup_length):
        """Linear rampup"""
        assert current >= 0 and rampup_length >= 0
        if current >= rampup_length:
            return 1.0
        else:
            return current / rampup_length

    @staticmethod
    def cosine_rampdown(current, rampdown_length):
        """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
        assert 0 <= current <= rampdown_length
        return max(0., float(.5 * (np.cos(np.pi * current / rampdown_length) + 1)))

def cycling_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch,args):

    lr = optimizer.param_groups[0]['lr']

    epoch = epoch + step_in_epoch / total_steps_in_epoch

    if epoch >= args.epochs and args.num_cycles>0 :
        lr =  args.initial_lr
        lr *= ramps.cosine_rampdown((epoch - args.epochs) % args.cycle_interval,args.cycle_interval)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def SD_cosine_annealing(step, interval, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + np.cos(step % interval / interval * np.pi))



if __name__ == "__main__":
    lr_list = []
    for i in range(160):
        lr_list.append(SD_cosine_annealing(i,40,0.1,0))


    import matplotlib.pyplot as plt

    ax=plt.figure()
    plt.plot(lr_list)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.xlabel("迭代次数")
    plt.ylabel("学习率")
    plt.title("cosine 循环学习率示意图")
    plt.show()
    '''
    class arg(object):
        def __init__(self):
            self.lr_rampup=0
            #self.base_lr = 0.1
            self.lr=0.1
            self.initial_lr=0.01
            self.epochs=180
            self.lr_rampdown_epochs=210
            self.constant_lr_epoch=0
            self.cycle_interval=30
            self.constant_lr = None
            self.cycle_rampdown_epochs=30

            self.start_epoch = 0
            self.num_cycles = 20
            self.schedule = [80,120,160]
            self.gamma = 0.1


    args = arg()

    import torch

    model = torch.nn.Sequential(torch.nn.Linear(10,10))
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=0.01,
                                weight_decay=1e-4)

    lr_list = []
    for epoch in range(args.start_epoch,args.epochs+ args.num_cycles*args.cycle_interval + 1):
        if epoch in args.schedule:
            lr = optimizer.param_groups[0]['lr']
            lr *= args.gamma
            print('\n Epoch:{} LR decay:{}'.format(epoch, lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for i in range(391):
            cycling_learning_rate(optimizer,epoch,i,391,args)

        lr_list.append(optimizer.param_groups[0]['lr'])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(lr_list)
    #plt.yscale('log')
    plt.savefig('lr.png')
    plt.show()
    '''

# -*- coding: utf-8 -*-
# @Time    : 2019/4/10 22:26

# @Author  : ChengBin

# @Desc : ==============================================

# ======================================================

# @Project : Label_embedding

# @FileName: test_function.py.py

# @Software: PyCharm

import  torch
import numpy as np

def cosine_annealing(step, interval, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + np.cos(step % interval / interval * np.pi))


'''
model = torch.nn.Sequential(torch.nn.Linear(10,10))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: cosine_annealing(
                    step,40,
                    1,
                    0))

lr =[]
for i in range(180):
    scheduler.step()
    lr.append(scheduler.get_lr()[0])

import matplotlib.pyplot as plt
plt.plot(lr)
plt.show()
'''


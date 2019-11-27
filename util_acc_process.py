# -*- coding: utf-8 -*-
# @Time    : 2019/8/27 11:44

# @Author  : ChengBin

# @Desc : ==============================================

# ======================================================

# @Project : Label_embedding

# @FileName: util_acc_process.py

# @Software: PyCharm
import sys
sys.path.append("..")
#import config.configer
from config import configer
import os
import torch
from utils import AverageMeter,accuracy,Bar


config = configer()
#path = sys.argv[1]
path = "final_result/cifar100/densenet100/densenet-depth-100-growthRate-12-epochs-300-milestones-150225/densenet-depth-100-growthRate-12-epochs-300-milestones-150225-T1/"

args = config.load_args_from_path(path)
model = config.create_model(args)
trainloader, testloader=config.get_loader(args)
train_criterion, test_criterion = config.get_criterion()


model = torch.nn.DataParallel(model).cuda()
checkfilename = 'pth'

pathDir = os.listdir(path)      #获取当前路径下的文件名，返回List
pathDir.sort()


record = torch.zeros(10000).cuda()
for file in pathDir:
    if (checkfilename in file):
        checkpoint = torch.load(path+file)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        #model = torch.nn.DataParallel(model).cuda()
        #print(checkpoint['state_dict'])
        model.load_state_dict(checkpoint['state_dict'])
        #model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
        model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        bar = Bar('Processing', max=len(testloader))
        bar.width = 16
        print(file)

        for batch_idx, datainfo in enumerate(testloader):
            datalen = datainfo.__len__()
            if datalen == 3:
                inputs, targets, idx = datainfo
            elif datalen == 5:
                inputs, targets, noisy_prob, clean_prob, idx = datainfo
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs, requires_grad=False), torch.autograd.Variable(targets)
            outputs = model(inputs)
            loss = test_criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5), args=args, criterion=test_criterion)
            batch = targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            tmp = (predicted == targets).view(batch).type(torch.cuda.FloatTensor)
            tmp2 = record[idx]

            record[idx] = tmp2+ tmp


            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))


            bar.suffix = '({batch}/{size}) | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                file = file,
                batch=batch_idx + 1,
                size=len(testloader),
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()

        print('\n' ,record)
        bar.finish()




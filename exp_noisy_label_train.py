from __future__ import print_function
import os
import torch
import argparse
import random
from config import configer
from utils import mkdir_p,Logger,AverageMeter,accuracy
from progress.bar import Bar 

import torch.backends.cudnn as cudnn
import time

start = time.time()

import sys
path = sys.executable
print(path)

parser = argparse.ArgumentParser(description='PyTorch Training')

### edit the config.py to add argument
config = configer()

args = config.add_argument(parser)


if args.config_path:
    args=config.load_args_from_path(args.config_path,args)


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


args = config.get_num_classes_and_out_dir()

if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

state = {k: v for k, v in args._get_kwargs()}

#record = config.get_record(sys.argv)
config.save_setting(state,os.path.join(args.checkpoint,'config.json'))
file_handle = open(os.path.join(args.checkpoint,'command.txt'), mode='w')
file_handle.write(sys.argv)


best_acc = 0

print(config.args.checkpoint)

def main():

    global best_acc

    # data
    print('==> Preparing dataset %s' % args.dataset)
    trainloader, testloader=config.get_loader(args)

    # Model
    print("==> creating model '{}'".format(args.arch))

    model = config.create_model()
    #model = torch.nn.Sequential(torch.nn.Linear(3,3))

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))


    tfboard_dir = []
    if args.tfboard_dir:
        tfboard_dir = args.tfboard_dir
    else:
        tfboard_dir = args.checkpoint

    train_criterion, test_criterion = config.get_criterion()

    optimizer, scheduler = config.get_optimizer(model)



    title = args.arch + str(args.depth)
    if args.resume:
        # Load checkpoint
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), tfboard_dir=tfboard_dir,title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), tfboard_dir=tfboard_dir,title=title)
        logger.set_names(['learning_rate', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, test_criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    print("whole epoch:{}".format(args.epochs))

    start_epoch = args.start_epoch

    for epoch in range(start_epoch,args.epochs):

        scheduler.step()
        print('\nEpoch: [%d | %d] LR: %f dir: %s' %
              (epoch + 1, args.epochs, scheduler.get_lr()[0], args.checkpoint))

        train_loss, train_acc = train(trainloader, model, train_criterion, optimizer, epoch, use_cuda,args)
        test_loss, test_acc = test(testloader, model, test_criterion, epoch, use_cuda)

        logger.append([scheduler.get_lr()[0], train_loss, test_loss, train_acc, test_acc])

        info = {'train_acc': train_acc, \
                'test_acc': test_acc, \
               'train_loss': train_loss, \
                'test_loss': test_loss, \
                'learning_rate': scheduler.get_lr()[0]}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        config.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, epoch,is_best,train_criterion=train_criterion,test_criterion=test_criterion,checkpoint=args.checkpoint,args=args)





    logger.close()
    args.Finished = True
    args.best_acc = best_acc

    global start
    args.exp_time = time.time()-start
    state = {k: v for k, v in args._get_kwargs()}
    config.save_setting(state, os.path.join(args.checkpoint, 'config.json'))
    config.save_experiment(state)
    config.save_acc(best_acc, args.checkpoint)
    print('Best acc: {}'.format(best_acc))

    logger.plot(path=args.checkpoint)

def train(trainloader, model, train_criterion, optimizer, epoch, use_cuda,args):
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(trainloader))
    bar.width =16

    for batch_idx, datainfo in enumerate(trainloader):
        datalen=datainfo.__len__()
        if datalen==3:
            inputs, targets , idx = datainfo
        elif datalen == 5 :
            inputs, targets, noisy_prob, clean_prob, idx = datainfo
        if use_cuda:
            inputs, targets = train_criterion.prepare(inputs,targets)
        #print(inputs.size(),targets.size())


        outputs = model(inputs)

        loss = train_criterion(outputs,targets,idx,epoch)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5),args=args,criterion=train_criterion)

        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        total=time.time() - start

        bar.suffix = '({batch}/{size}) | ETA: {eta:} | Total: {total:.2f} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            total=total,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg
        )
        bar.next()

    train_criterion.process(epoch,args)

    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):

    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    bar = Bar('Processing', max=len(testloader))
    bar.width = 16
    for batch_idx, datainfo in enumerate(testloader):
        datalen = datainfo.__len__()
        if datalen == 3:
            inputs, targets, idx = datainfo
        elif datalen == 5:
            inputs, targets, noisy_prob, clean_prob, idx = datainfo

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, requires_grad=False), torch.autograd.Variable(targets)

        outputs = model(inputs)

        loss = criterion(outputs,targets)

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5),args=args,criterion=criterion)
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        total = time.time() - start
        bar.suffix = '({batch}/{size}) | ETA: {eta:} | Total: {total:.2f} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            eta=bar.eta_td,
            total = total,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()

    bar.finish()
    return (losses.avg, top1.avg)

if __name__ == '__main__':
    main()













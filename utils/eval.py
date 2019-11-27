from __future__ import print_function, absolute_import
import torch

__all__ = ['accuracy','accuracy_plus']

def accuracy(output, target, topk=(1,),args=None,criterion=None):
    """Computes the precision@k for the specified values of k"""

    if args is not None:
        if args.use_metric_label:
            if args.update_type == 'ensemble':
                output=criterion.Euclid(criterion.processor.emsemble_label, output)


    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))


    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_plus(outputs, targets, classnum):
    """Computes the precision@k for the specified values of k"""
    batch = targets.size(0)

    _, predicted = torch.max(outputs.data, 1)
    index = (predicted == targets)
    select = index.view(batch,1).type(torch.cuda.FloatTensor)
    acc = index.sum().item()*100/batch


    feature_num = outputs.size(1)
    mask = torch.zeros(batch, classnum).cuda()

    mask=mask.type(torch.cuda.FloatTensor) * select

    mask = mask.scatter_(1, targets.view(batch, 1), 1).cuda()  # N*k 每一行是一个one-hot向量

    mask = mask.view(batch, classnum, 1)  # N*k*1 目的是扩成 N*k*s
    soft_logit = torch.nn.functional.softmax(outputs, dim=1)
    output_ex = soft_logit.view(batch, 1, feature_num)  # N*1*s 目的是扩成 N*k*s
    sum = torch.sum(output_ex * mask, dim=0)


    return acc,sum,mask
from __future__ import absolute_import

'''Resnet for cifar dataset.
The base model is ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py 
'''

import torch.nn as nn
import math
import math
from torch.nn import Parameter
import torch

__all__ = ['resnet','resnet_mapping','resnet_2fc']


# region 1_base_resnet_region
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

### Origin resnet
class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)


        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
# endregion

# region 2_learnable_label_resnet_region


class ResNet_no_fc(nn.Module):
    def __init__(self, depth, feature_num):
        super(ResNet_no_fc, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >= 44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, feature_num)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

class ResNet_to_label(nn.Module):

    def __init__(self, depth, feature_num):
        super(ResNet_to_label, self).__init__()
        self.resnet_no_fc = ResNet_no_fc(depth, feature_num)
        self.fc = self.resnet_no_fc.fc

    def forward(self, x):
        x = self.resnet_no_fc(x)
        x = self.fc(x)
        return x

'''
class comdist():
    @staticmethod
    ## compute the Euclidean distance between label W and feature x
    def Euclid(W,x):
        batch_size = x.size(0)
        num_classes = W.size(0)
        dists = torch.zeros(num_classes, batch_size).cuda()
        dists += torch.sum(x ** 2, dim=1).reshape(1, batch_size)
        dists += torch.sum(W ** 2, dim=1).reshape(num_classes, 1)
        dists -= 2 * W.mm(x.t())
        dists = torch.clamp(dists, min=0)
        dists = torch.sqrt(dists)
        dists = -1 * dists
        dists = dists.t()
        return dists
    ## compute the cosine distance between label W and feature x
    def Cosine(W,x):
        W_norm=W.norm(p=2,dim=1)
        x_norm=x.norm(p=2,dim=1)
        cos_theta=x.mm(W.t())/(W_norm*x_norm.view(-1,1))
        cos_theta = cos_theta.clamp(-1, 1)
        return cos_theta
    ## directly print x
    def Dx(W,x):
        return x


class activateF():
    @staticmethod
    # add an Identity Function between label W and feature x
    def Identity(x):
        return x
    # add a tanh Function between label W and feature x
    def tanh(x):
        return torch.tanh(x)
    # add a relu Function between label W and feature x
    def relu(x):
        return torch.nn.funtional.relu(x)
'''

class mappinglayer(nn.Module):
    def __init__(self, depth, MaplabelInit, num_classes, feature_num, InitFactor=1,
                 Stable=False, Dmode='Euclid', Afun='Identity',Layeradjust = 'toLabel'):
        super(mappinglayer, self).__init__()

        if Stable==False:
            self.maplabel = Parameter(torch.Tensor(num_classes, feature_num))
            self.maplabel.data = MaplabelInit * InitFactor
            print("==> Label is learnable")
        else:
            self.maplabel = MaplabelInit.cuda() * InitFactor
            print("==> Label is stable")


        if Layeradjust =='toLabel':
            print('==> The fullyconnection layer to feature number')
            self.resnet_to_label = ResNet_to_label(depth, feature_num)
        elif Layeradjust =='noFc':
            print('==> Do no use fullyconnection layer')
            self.resnet_to_label = ResNet_no_fc(depth, feature_num)
        elif Layeradjust =='base':
            print('==> The fullyconnection layer to class number')
            self.resnet_to_label = ResNet_to_label(depth, num_classes)

        self.num_classes = num_classes

        if Dmode == 'Euclid':
            print("==> Compute the Euclidean distance between label and feature ")
            self.com_score =self.Euclid
        elif Dmode == 'Cosine':
            print("==> Compute the Cosine distance between label and feature ")
            self.com_score = self.Cosine
        elif Dmode == 'Dx':
            self.com_score = self.Dx
            print("==> Directly compute x")

        if InitFactor!=1:
            print('InitFactor:'+str(InitFactor))

        if Afun == 'Identity':
            self.af = self.Identity
            print("==> No activation function between label and feature")
        elif Afun == 'tanh':
            self.af = self.tanh
            print("==> Add a tanh Function between label and feature")
        elif Afun == 'relu':
            self.af = self.relu
            print("==> Add a relu Function between label and feature")


    def forward(self, x):
        x = self.resnet_to_label(x)
        x = self.af(x)
        W = self.maplabel
        distance=self.com_score(W,x)
        return distance

    def Euclid(self,W,x):
        batch_size = x.size(0)
        num_classes = W.size(0)
        dists = torch.zeros(num_classes, batch_size).cuda()
        dists += torch.sum(x ** 2, dim=1).reshape(1, batch_size)
        dists += torch.sum(W ** 2, dim=1).reshape(num_classes, 1)
        dists -= 2 * W.mm(x.t())
        dists = torch.clamp(dists, min=0)
        dists = torch.sqrt(dists)
        dists = -1 * dists
        dists = dists.t()
        return dists

    def Cosine(self,W,x):
        W_norm=W.norm(p=2,dim=1)
        x_norm=x.norm(p=2,dim=1)
        cos_theta=x.mm(W.t())/(W_norm*x_norm.view(-1,1))
        cos_theta = cos_theta.clamp(-1, 1)
        return cos_theta

    def Dx(self,W,x):
        return x

    def Identity(self,x):
        return x

    def tanh(self,x):
        return torch.tanh(x)

    def relu(self,x):
        return torch.nn.funtional.relu(x)





def resnet_mapping(**kwargs):
    #Constructs a learnable_label_ResNet model.
    return mappinglayer(**kwargs)

# endregion

# region 3_two_fullyconnection_layer_resnet_region
class ResNet_two_fc(nn.Module):

    def __init__(self, depth, feature_num,class_num):
        super(ResNet_two_fc, self).__init__()
        self.resnet_no_fc = ResNet_no_fc(depth, feature_num)
        self.fc1 = self.resnet_no_fc.fc
        self.fc2 = nn.Linear(feature_num,class_num)

    def forward(self, x):
        x = self.resnet_no_fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def resnet_2fc(**kwargs):
    """
    Constructs a two_fullyconnection_layer ResNet model.
    """
    return ResNet_two_fc(**kwargs)

# endregion










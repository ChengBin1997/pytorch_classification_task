# -*- coding: utf-8 -*-
import models.cifar as models
import json
import shutil
import torchvision.transforms as transforms
import torch.utils.data as data
import sys
sys.path.append("../")
import os
from method import *
from utils import *
from utils.data_process import noisy_cifar10,noisy_cifar100
from utils.LR_adjust import SD_cosine_annealing


class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, object):
            state = obj.__class__.__name__
            return state
        return json.JSONEncoder.default(self, obj)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class configer(object):
    def __init__(self):
        self.args=[]
        self.num_classes=[]
        self.best_acc = 0
        self.model=[]


    def save_setting(self,model_setting,path):


        config_file = json.dumps(model_setting,sort_keys=True, indent=4, separators=(',', ':'),cls=DateEncoder)
        with open(path, 'w') as json_file:
            json_file.write(config_file)

    def load_setting(self,path):
        with open(path, 'r') as json_file:
            return json.load(json_file)

    def save_checkpoint(self,state, epoch, is_best, train_criterion,test_criterion,checkpoint, args, filename='checkpoint.pth.tar'):
        filepath = os.path.join(checkpoint, filename)

        if args.clTE:
            state['train_ensemble_label']=train_criterion.processor.emsemble_label
        elif args.slTE:
            state['train_ensemble_label'] = train_criterion.processor.emsemble_label
            if args.process_type == 'dirichlet':
                state['D_parm'] = train_criterion.processor.D_parm




        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

        is_save = False
        if (epoch <= args.epochs) and (epoch % 20 == 0):
            is_save = True

        if is_save:
            print('save the checkpoint!')
            shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint-e'+str(epoch)+'.pth.tar'))




    #####参数配置里有一个坑，存储的学习率是最后的学习率，要还原实验的话需要改
    def save_experiment(self, model_setting, path=None):
        if path ==None:
            path = 'final_result/all.json' if model_setting['is_final'] else 'test_result/all.json'
        total = []
        if not os.path.exists(path):
            os.mknod(path)
            saver = {}
            saver['total'] = 1
            saver['experiment'] = []
            saver['experiment'].append(model_setting)
            with open(path, 'w') as json_file:
                saver_file = json.dumps(saver, ensure_ascii=False, sort_keys=True, indent=4, separators=(',', ':'),cls=DateEncoder)
                json_file.write(saver_file)
        else:
            with open(path, 'r') as json_file:
                saver = json.load(json_file)
            with open(path, 'w') as json_file:
                saver['total'] = saver['total'] + 1
                saver['experiment'].append(model_setting)
                saver_file = json.dumps(saver, ensure_ascii=False, sort_keys=True, indent=4, separators=(',', ':'),cls=DateEncoder)
                json_file.write(saver_file)

    def save_acc(self,best_acc, pathDir):
        filename = os.path.join(pathDir, 'best_acc.txt')
        file_handle = open(filename, mode='w')
        file_handle.write(str(best_acc))


    ### To add other model,you should edit the create_model and add argument method
    def create_model(self,args = None):
        if args == None:
            args = self.args

        self.num_classes = args.num_classes
        if args.arch.startswith('resnext'):
            model = models.__dict__[args.arch](
                cardinality=args.cardinality,
                num_classes=self.num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
            )
        elif args.arch.startswith('densenet'):
            model = models.__dict__[args.arch](
                num_classes=self.num_classes,
                depth=args.depth,
                growthRate=args.growthRate,
                compressionRate=args.compressionRate,
                dropRate=args.drop,
            )
        elif args.arch.startswith('efficient_densenet'):
            model = models.__dict__[args.arch](
                growth_rate=args.growthRate,
                num_classes=self.num_classes,
                depth=args.depth,
                small_inputs=True,
                efficient=True,
                drop_rate = args.drop,
                compression = args.compressionRate,
            )
        elif args.arch.startswith('wrn'):
            model = models.__dict__[args.arch](
                num_classes=self.num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
            )
        elif  args.arch.endswith('resnet'):
            model = models.__dict__[args.arch](
                num_classes=self.num_classes,
                depth=args.depth,
            )
        elif args.arch.endswith('reslabelnet'):
            model = models.__dict__[args.arch](
                num_classes=self.num_classes,
                depth=args.depth,
            )
        elif args.arch.endswith('shake_shake'):
            model = models.__dict__[args.arch](args)
        elif args.arch.endswith('lenet'):
            model = models.__dict__[args.arch](args=args)
        elif args.arch.endswith('mlp'):
            model = models.__dict__[args.arch](args=args)
        else:
            model = models.__dict__[args.arch](num_classes=self.num_classes)


        if args.use_metric_label and args.update_type=='backpropagation':
            print('finetune the model')
            model = metric_label_model(model,args.num_classes,args.metric_label_type)

        print(model)
        self.model=model
        return model


    def add_argument(self,parser):
        model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))
        # Datasets
        parser.add_argument('-d', '--dataset', default='cifar10', type=str)
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        # Optimization options
        parser.add_argument('--epochs', default=180, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('--train_batch', default=128, type=int, metavar='N',
                            help='train batchsize')
        parser.add_argument('--test_batch', default=100, type=int, metavar='N',
                            help='test batchsize')
        parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--drop', '--dropout', default=0, type=float,
                            metavar='Dropout', help='Dropout ratio')
        parser.add_argument('--milestones', type=int, nargs='+', default=[80, 120, 160],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument(
            '--scheduler',
            type=str,
            default='multistep')
        parser.add_argument('--T_max',type=int,default=30)
        parser.add_argument('--eta_min', type=float, default=0.0)

        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
        # Checkpoints
        parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='PATH',
                            help='path to save checkpoint (default: checkpoint)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        # Architecture
        parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                            choices=model_names,
                            help='model architecture: ' +
                                 ' | '.join(model_names) +
                                 ' (default: resnet18)')
        parser.add_argument('--depth', type=int, default=0, help='Model depth.')
        parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
        parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
        parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
        parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
        # Miscs
        parser.add_argument('--manualSeed', type=int, help='manual seed')
        parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
        # Device options
        parser.add_argument('--gpu_id', default='0', type=str,
                            help='id(s) for CUDA_VISIBLE_DEVICES')

        # Parameter about leanable label
        parser.add_argument('--featureNum', type=int, help='feature number')
        parser.add_argument('--MaplabelInit', type=str, default='random',
                            help='Initialization tensor of the label:random,one_hot or hadamard')
        parser.add_argument('--b', type=float, default=1, help='softmax factor')
        parser.add_argument('--Ifactor', type=float, default=1, help='Init factor')

        parser.add_argument('--tfboard_dir', type=str, help='tensorboard dir')
        parser.add_argument('--stable', type=bool, default=False)

        #  Temporal ensemble

        parser.add_argument('--clTE', action="store_true",help='Category level Temporal ensemble')
        parser.add_argument('--clTE_update_interval',type=int,default=1)
        parser.add_argument('--clTE_start_epoch', type=int, default=0)
        parser.add_argument('--is_select', action="store_true", help='If use the select setting')
        parser.add_argument('--R', type=float, default=0.01, help='Emsemble rate')
        parser.add_argument('--P', type=float, default=0.5, help='the average output : one-hot is p:(1-p) ')
        #parser.add_argument('--TE_start_epoch', '-se',type=int, default=0, help='When temporal ensemble start')
        parser.add_argument('--updatetype',type=int,default=0,help="when to update the label")
        parser.add_argument('--model_type', type=str, default='simple')
        parser.add_argument('--loss_type', type=str, default='CE')
        parser.add_argument('--namda', type=float, default=1, help='loss rate')
        parser.add_argument('--process_type', type=str, default='normal')
        parser.add_argument('--init_factor', type=int, default=1000, help='init factor')

        parser.add_argument('--AclTE',action="store_true")
        parser.add_argument('--Alternate_interval', type=int,default=30)
        parser.add_argument('--Alternate_rate', type=float, default=0.5)
        parser.add_argument('--prior_type', type=str,choices=['one_hot', 'history', 'both'])



        parser.add_argument('--slTE', action="store_true", help='Sample level Temporal ensemble')
        parser.add_argument('--slTE_update_interval', type=int, default=1)
        parser.add_argument('--slTE_start_epoch', type=int, default=0)




        # learning rate adjusting
        parser.add_argument('--initial_lr', default=0.01, type=float,
                            metavar='LR', help='initial learning rate when using linear rampup')
        parser.add_argument('--cycle_interval', default=4, type=int,
                            help='the number of epochs for small cyclical learning rate')
        parser.add_argument('--num_cycles', default=0, type=int, help='If 1 or more, use additional cycles after args.epochs')

        # uncertain label
        parser.add_argument('--uctl', action="store_true", help='uncertain label')

        # use regular label processor
        parser.add_argument('--use_rgl_sl_lp', action='store_true')
        parser.add_argument('--rgl_type', type=str, default='even')
        parser.add_argument('--rgl_interval', type=int, default=5)

        # use label smoothing
        parser.add_argument('--use_label_smoothing', action='store_true')
        parser.add_argument('--label_smoothing_epsilon', type=float, default=0.1)

        parser.add_argument('--base_channels', type=int)
        parser.add_argument('--shake_forward', action='store_true')
        parser.add_argument('--shake_backward', action='store_true')
        parser.add_argument('--shake_image',action='store_true')

        # train mode
        parser.add_argument('--is_final',action = 'store_true')

        # noisy data set
        parser.add_argument('--noise_type',type=str,default='none',
                            choices=['none', 'asymmetric', 'symmetric'])
        parser.add_argument('--noise_ratio', type=float, default=0.0)

        # TCR
        parser.add_argument('--use_TCR',action = 'store_true')
        parser.add_argument('--TCR_start_epoch',type=int,default=1)
        parser.add_argument('--TCR_stop_epoch', type=int,default=300)
        parser.add_argument('--TCR_beta', type=float,default=0.1)
        parser.add_argument('--TCR_squeeze_ratio', type=float, default=1.1)
        parser.add_argument('--start_squeeze', type=int, default=80)

        # snapshot
        parser.add_argument('--use_snap_shot',action = 'store_true')
        parser.add_argument('--use_bayesian_snap_shot', action='store_true')
        parser.add_argument('--SD_interval', type=int,default=41)
        parser.add_argument('--SD_T', type=int, default=4)

        # Penalizing_Confident_criterion
        parser.add_argument('--use_penalizing_output',action='store_true')

        parser.add_argument('--use_label_attention', action='store_true')


        parser.add_argument('--use_metric_label', action='store_true')
        parser.add_argument('--metric_label_type', type=str, default='one_hot',
                            choices=['one_hot', 'hadamard'])
        parser.add_argument('--update_type', type=str, default='backpropagation',
                            choices=['backpropagation', 'ensemble'])
        parser.add_argument('--feature_num', type=int)


        parser.add_argument('--use_label_select', action='store_true')

        parser.add_argument('--mergedataset', action='store_true')

        #conparser.add_argument('--load_config', action='store_true')
        parser.add_argument('--cfg', type=str,help='use for load config')

        parser.add_argument('--use_soft_label', action='store_true')
        parser.add_argument('--model_path', type=str, help='use for model_path')

        parser.add_argument('--use_anti_kld',action='store_true')

        parser.add_argument('--use_mix_up',action='store_true')
        parser.add_argument('--mix_up_alpha',type =float,default=1.0)

        parser.add_argument('--use_correct_mix_up', action='store_true')

        parser.add_argument('--use_variance_control',action='store_true')
        parser.add_argument('--vc_start_epoch', type=float, default=1)
        parser.add_argument('--vc_beta', type=float, default=1)
        parser.add_argument('--vc_is_count',action='store_true')

        parser.add_argument('--distill',action='store_true')
        parser.add_argument('--d_type', type=str, default='base')
        parser.add_argument('--t_path',type=str,default='',metavar='PATH')
        parser.add_argument('--t_model',type=str,default='')
        parser.add_argument('--lambda_st',type=float,default=0.1)
        parser.add_argument('--clsw',type=float,default=1.0)
        parser.add_argument('--T',type=float,default=3.0)


        args=parser.parse_args()

        self.args = args

        return args



    def get_num_classes_and_out_dir(self,args=None):
        if args == None:
            args = self.args

        assert args.dataset in [
            'cifar10', 'cifar100', 'MNIST', 'FashionMNIST', 'KMNIST','svhn'
        ]

        if args.is_final:
            main_dir = 'final_result'
        else:
            main_dir = 'test_result'

        if args.mergedataset:
            main_dir = main_dir+'/merge_dataset'

        if args.dataset == 'cifar10':
            num_classes = 10
            if args.depth != 0:
                checkpointdir = main_dir+'/cifar10/' + args.arch + str(args.depth) + '/'
            else:
                checkpointdir = main_dir+'/cifar10/' + args.arch + '/'
        elif args.dataset == 'cifar100':
            num_classes = 100
            if args.depth != 0:
                checkpointdir = main_dir+'/cifar100/' + args.arch + str(args.depth) + '/'
            else:
                checkpointdir = main_dir+'/cifar100/' + args.arch + '/'
        elif args.dataset == 'svhn':
            num_classes = 10
            if args.depth != 0:
                checkpointdir = main_dir+'/svhn/' + args.arch + str(args.depth) + '/'
            else:
                checkpointdir = main_dir+'/svhn/' + args.arch + '/'
        elif args.dataset == 'MNIST':
            num_classes = 10
            if args.depth != 0:
                checkpointdir = main_dir+'/mnist/' + args.arch + str(args.depth) + '/'
            else:
                checkpointdir = main_dir+'/mnist/' + args.arch + '/'


        args.num_classes = num_classes
        self.num_classesm = num_classes

        import sys
        state = {k: v for k, v in args._get_kwargs()}
        dir= checkpointdir + self.get_dir(state, sys.argv)
        dir = dir + '-T'
        index = 1
        while (True):
            if not os.path.isdir(dir + str(index)):
                args.checkpoint = dir + str(index)
                break
            else:
                index += 1

        args.Finished = False

        return args


    def get_dir(self,state, argv):

        parm = (' '.join(argv))
        #parm = parm.replace('--', '-')
        arg = parm.split('--')

        configkey = list(state.keys())+['wd']

        dir = state['arch']

        for setting in arg:
            key = setting.split(' ')[0]

            if key in configkey and key not in ['dataset', 'arch', 'checkpoint','is_final','model_path','config_path','gpu_id','cfg','t_path']:
                dir = dir+'-' + key
                rest = setting.replace(key, '').replace(' ', '')
                if rest is not '':
                    dir = dir + '-' + rest





        dir = dir+'/'+dir
        dir.replace('\r', '').replace('\n', '')
        return dir

    def get_record(self,argv):
        parm = (' '.join(argv[1:]))
        parm = parm.replace('-', '--')
        arg = parm.split('--')
        print(arg)
        record = dict()
        for setting in arg:
            key = setting.split(' ')[0]
            rest = setting.replace(key, '').replace(' ', '')
            if key != "":
                record[key] =rest

        return record



    def get_loader(self,args=None):

        #print('==> Preparing dataset %s' % args.dataset)
        if args.dataset == 'cifar10' or args.dataset =='cifar100':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            if args.noise_type == 'none':
                print('==> use clean dataset')
                if args.dataset == 'cifar10':
                    dataset = scifar10
                else:
                    dataset = scifar100
                trainset = dataset(root='./data', train=True, download=True, transform=transform_train)
                testset = dataset(root='./data', train=False, download=False, transform=transform_test)

                if args.mergedataset==True:
                    trainset = torch.utils.data.ConcatDataset([trainset, testset])

                trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers,pin_memory=True)
                testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers,pin_memory=True)
            else:
                print('==> use dataset with noisy label')
                if args.dataset == 'cifar10':
                    dataset = noisy_cifar10
                else:
                    dataset = noisy_cifar100

                trainset = dataset(noise_type=args.noise_type,
                                                  noise_ratio=args.noise_ratio,
                                                  root='./data', train=True, download=True, transform=transform_train)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
                testset = dataset(root='./data', train=False, download=True,
                                                       transform=transform_test)
                testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        elif args.dataset == 'svhn':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            dataset = sSVHN
            trainset = dataset(root='~/.torchvision/datasets/SVHN', split='train', download=True, transform=transform_train)
            testset = dataset(root='~/.torchvision/datasets/SVHN', split='test', download=True, transform=transform_test)

            trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
            testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        elif args.dataset == 'MNIST':
            transform_train = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
            dataset = sMnist
            trainset = dataset(root='./data', train=True, download=True, transform=transform_train)
            testset = dataset(root='./data', train=False, download=False, transform=transform_test)
            trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
            testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


        self.args.num_samples = len(trainset)
        print('trainset:{}'.format(len(trainset)))

        self.args.trainloder = trainloader
        self.args.testloader = testloader

        return trainloader,testloader



    def get_criterion(self,args=None):
        if args == None:
            args = self.args
        if args.clTE:
            train_criterion = CategoryLabelProcessCriterion(args.num_classes,args.R,args.P,args.is_select,
                                                            args.model_type,args.loss_type,args.namda,args.process_type,args.init_factor)
        elif args.AclTE:
            train_criterion = AlternateCategoryLabelProcessCriterion(args.num_classes,args.R,args.P,args.is_select,
                                                            args.model_type,args.loss_type,args.namda,args.process_type,args.init_factor,
                                                                args.Alternate_interval,args.Alternate_rate,args.is_no_one_hot)
        elif args.slTE:
            train_criterion = SampleLabelProcessCriterion(args)
        elif args.use_label_smoothing:
            train_criterion = label_smoothing_criterion(args.label_smoothing_epsilon,args.num_classes)
        elif args.use_TCR:
            train_criterion = TCRCriterion(self.args.num_samples,args.num_classes,args.TCR_start_epoch,
                                           args.TCR_beta,args.start_squeeze,args.TCR_squeeze_ratio,args.TCR_stop_epoch)
        elif args.use_snap_shot:
            train_criterion = SnapshotCriterion(args.num_samples,args.num_classes,args.SD_T,args.SD_interval)
        elif args.use_bayesian_snap_shot:
            train_criterion = beyesianSnapshotCriterion(args)
        elif args.use_penalizing_output:
            train_criterion = Penalizing_Confident_criterion()
        elif args.use_metric_label:
            if args.update_type == 'backpropagation':
                train_criterion = metric_label_criterion(self.model)
            elif args.update_type == 'ensemble':
                train_criterion = metric_label_ensemble_criterion(self.model,args.num_classes,
                                                     args.metric_label_type,args.feature_num,args.R,args.P,args.is_select)
        elif args.use_label_select:
            train_criterion = label_select_criterion(args.num_samples,args.num_classes)
        elif args.use_soft_label:
            init_label=self.get_label_from_model(args.config_path,args)
            train_criterion = soft_label_criterion(init_label)
        elif args.use_anti_kld:
            train_criterion=anti_kld_Criterion(args.label_smoothing_epsilon,args.num_classes)
        elif args.use_mix_up:
            train_criterion= mix_up_criterion(args.mix_up_alpha)
        elif args.use_correct_mix_up:
            train_criterion= correct_mix_up_criterion(args.mix_up_alpha)
        elif args.use_variance_control:
            train_criterion= VarianceControlCriterion(self.args.num_samples,args.num_classes,args.vc_start_epoch,args.vc_beta,args.vc_is_count)
        elif args.distill:
            print('Teacher model path:{}'.format(args.t_path))
            print('student lambda:{},classification weight:{},T:{}'.format(args.lambda_st,args.clsw,args.T))
            this_config = configer()
            t_model = this_config.create_model_by_args(args.t_path+'/config.json')
            #t_model = t_model.to('cuda')

            checkpoint = torch.load(args.t_path + '/model_best.pth.tar')
            t_model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
            t_model = torch.nn.DataParallel(t_model).cuda()
            train_criterion =ditill_criterion(t_model,args.lambda_st,args.T,args.d_type,args.clsw,args.num_classes)
        else:
            train_criterion = cross_entropy_criterion()

        test_criterion = torch.nn.CrossEntropyLoss()
        return train_criterion,test_criterion


    def get_label_from_model(self,path,args):
        train_loder = self.args.trainloder
        last_model_args = self.load_args_from_path(path)
        model = self.create_model(last_model_args)
        checkpoint=torch.load(path + '/model_best.pth.tar')
        model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
        num_samples = args.num_samples
        soft_label = torch.zeros(num_samples,args.num_classes)
        print(soft_label.size())

        model.eval()

        for batch_idx, datainfo in enumerate(train_loder):
            datalen = datainfo.__len__()
            if datalen == 3:
                inputs, targets, idx = datainfo
            elif datalen == 5:
                inputs, targets, noisy_prob, clean_prob, idx = datainfo

            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda(async=True)
                model =model.cuda()

            #inputs, targets = torch.autograd.Variable(inputs, requires_grad=False), torch.autograd.Variable(targets)
            #print(inputs.size())
            outputs = model(inputs)
            preds = torch.nn.functional.softmax(outputs,dim=1)
            soft_label[idx,:]=preds.cpu().data

        return soft_label







    def get_optimizer(self,model,args=None):
        if args == None:
            args = self.args

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)

        if args.scheduler=='multistep':
            scheduler= torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=args.milestones, gamma=args.gamma)
        elif args.scheduler=='cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,args.T_max, args.eta_min)
        elif args.scheduler == 'snap_shot':
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: SD_cosine_annealing(
                    step,args.SD_interval,
                    1,
                    0))


        return optimizer,scheduler


    def load_args_from_path(self,path,args=None):
        with open(path, 'r') as json_file:
            config_parm = json.load(json_file)

        if args==None:
            import argparse
            parser = argparse.ArgumentParser()
            args=self.add_argument(parser)

        for key, value in config_parm.items():
            setattr(args, key, value)

        # 控制台参数赋值
        '''
        import sys
        parm = (' '.join(sys.argv))
        input_arg = parm.split('--')
        key_list=[]
        for setting in input_arg:
            key = setting.split(' ')[0]
            key_list.append(key)

        print('key_list:{}'.format(key_list))

        for key, value in config_parm.items():
            if key not in key_list:
                setattr(args, key, value)
        '''

        #self.args=args
        #print(args)

        return args

    def create_model_by_args(self, path):
        #with open(path, 'r') as json_file:
        #    parm = json.load(json_file)
        #import argparse
        #parser = argparse.ArgumentParser()
        #args = parser.parse_args()
        args=self.load_args_from_path(path)
        #for key, value in parm.items():
        #    setattr(args, key, value)
        #print(args)

        num_classes = []
        if args.dataset == 'cifar10':
            num_classes = 10
        elif args.dataset == 'cifar100':
            num_classes = 100

        model = self.create_model(args)
        return model



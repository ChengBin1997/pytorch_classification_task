cd ..
python exp_train.py -a resnet --depth 20 --distill --t_path /data/chengbin/final_result/cifar10/resnet20/resnet-depth-20/resnet-depth-20-m0300/ --t_model resnet20 --d_type mixup --lambda_st 1
python exp_train.py -a resnet --depth 20 --distill --t_path /data/chengbin/final_result/cifar10/resnet20/resnet-depth-20/resnet-depth-20-m0300/ --t_model resnet20 --d_type mixup --lambda_st 10
python exp_train.py -a resnet --depth 20 --distill --t_path /data/chengbin/final_result/cifar10/resnet20/resnet-depth-20/resnet-depth-20-m0300/ --t_model resnet20 --d_type mixup --T 1.0
python exp_train.py -a resnet --depth 20 --distill --t_path /data/chengbin/final_result/cifar10/resnet20/resnet-depth-20/resnet-depth-20-m0300/ --t_model resnet20 --d_type mixup --T 0.3

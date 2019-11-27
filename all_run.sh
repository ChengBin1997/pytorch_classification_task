## cifar10
#python exp_train.py -a resnet --depth 20 --is_final --gpu-id 2
#python exp_train.py -a resnet --depth 20 --is_final --gpu-id 2 --use_label_smoothing
#python exp_train.py -a resnet --depth 20 --is_final --gpu-id 2 --clTE
#python exp_train.py -a resnet --depth 20 --is_final --gpu-id 2 --use_snap_shot --scheduler snap_shot
python exp_noisy_label_train.py -a resnet --depth 20 --is_final --gpu-id 2 --clTE --scheduler snap_shot

#python exp_train.py -a resnet --depth 44 --is_final --gpu-id 2
#python exp_train.py -a resnet --depth 44 --is_final --gpu-id 2 --use_label_smoothing
#python exp_train.py -a resnet --depth 44 --is_final --gpu-id 2 --clTE
#python exp_train.py -a resnet --depth 44 --is_final --gpu-id 2 --use_snap_shot --scheduler snap_shot
python exp_noisy_label_train.py -a resnet --depth 44 --is_final --gpu-id 2 --clTE --scheduler snap_shot

#python exp_train.py -a resnet --depth 110 --is_final --gpu-id 2
#python exp_train.py -a resnet --depth 110 --is_final --gpu-id 2 --use_label_smoothing
#python exp_train.py -a resnet --depth 110 --is_final --gpu-id 2 --clTE
python exp_noisy_label_train.py -a resnet --depth 110 --is_final --gpu-id 2 --use_snap_shot --scheduler snap_shot
python exp_noisy_label_train.py -a resnet --depth 110 --is_final --gpu-id 2 --clTE --scheduler snap_shot

python exp_noisy_label_train.py -a alexnet --is_final --gpu-id 2
python exp_noisy_label_train.py -a alexnet --is_final --gpu-id 2 --use_label_smoothing
python exp_noisy_label_train.py -a alexnet --is_final --gpu-id 2 --clTE
python exp_noisy_label_train.py -a alexnet --is_final --gpu-id 2 --use_snap_shot --scheduler snap_shot
python exp_noisy_label_train.py -a alexnet --is_final --gpu-id 2 --clTE --scheduler snap_shot

python exp_noisy_label_train.py -a vgg19_bn --is_final --gpu-id 2
python exp_noisy_label_train.py -a vgg19_bn --is_final --gpu-id 2 --use_label_smoothing
python exp_noisy_label_train.py -a vgg19_bn --is_final --gpu-id 2 --clTE
python exp_noisy_label_train.py -a vgg19_bn --is_final --gpu-id 2 --use_snap_shot --scheduler snap_shot
python exp_noisy_label_train.py -a vgg19_bn --is_final --gpu-id 2 --clTE --scheduler snap_shot

python exp_noisy_label_train.py -a preresnet --depth 110 --is_final --gpu-id 2
python exp_noisy_label_train.py -a preresnet --depth 110 --is_final --gpu-id 2 --use_label_smoothing
python exp_noisy_label_train.py -a preresnet --depth 110 --is_final --gpu-id 2 --clTE
python exp_noisy_label_train.py -a preresnet --depth 110 --is_final --gpu-id 2 --use_snap_shot --scheduler snap_shot
python exp_noisy_label_train.py -a preresnet --depth 110 --is_final --gpu-id 2 --clTE --scheduler snap_shot

python exp_noisy_label_train.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --milestones 150 225 --is_final --gpu-id 2
python exp_noisy_label_train.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --milestones 150 225 --is_final --gpu-id 2 --use_label_smoothing
python exp_noisy_label_train.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --milestones 150 225 --is_final --gpu-id 2 --clTE
python exp_noisy_label_train.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --milestones 150 225 --is_final --gpu-id 2 --use_snap_shot --sheduler snap_shot
python exp_noisy_label_train.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --milestones 150 225 --is_final --gpu-id 2 --clTE --scheduler snap_shot

python exp_noisy_label_train.py -a wrn --depth 28 --drop 0.3 --wd 5e-4 --gamma 0.2 --widen-factor 10 --is_final --gpu-id 2
python exp_noisy_label_train.py -a wrn --depth 28 --drop 0.3 --wd 5e-4 --gamma 0.2 --widen-factor 10 --is_final --gpu-id 2 --use_label_smoothing
python exp_noisy_label_train.py -a wrn --depth 28 --drop 0.3 --wd 5e-4 --gamma 0.2 --widen-factor 10 --is_final --gpu-id 2 --clTE
python exp_noisy_label_train.py -a wrn --depth 28 --drop 0.3 --wd 5e-4 --gamma 0.2 --widen-factor 10 --is_final --gpu-id 2 --use_snap_shot --scheduler snap_shot
python exp_noisy_label_train.py -a wrn --depth 28 --drop 0.3 --wd 5e-4 --gamma 0.2 --widen-factor 10 --is_final --gpu-id 2 --clTE --scheduler snap_shot

python exp_noisy_label_train.py -a shake_shake --depth 26 --base_channels 32 --shake_forward --shake_backward --shake_image --epoch 1800 --is_final --gpu-id 2
python exp_noisy_label_train.py -a shake_shake --depth 26 --base_channels 32 --shake_forward --shake_backward --shake_image --epoch 1800 --is_final --gpu-id 2 --use_label_smoothing
python exp_noisy_label_train.py -a shake_shake --depth 26 --base_channels 32 --shake_forward --shake_backward --shake_image --epoch 1800 --is_final --gpu-id 2 --clTE
python exp_noisy_label_train.py -a shake_shake --depth 26 --base_channels 32 --shake_forward --shake_backward --shake_image --epoch 1800 --is_final --gpu-id 2 --use_snap_shot --scheduler snap_shot
python exp_noisy_label_train.py -a shake_shake --depth 26 --base_channels 32 --shake_forward --shake_backward --shake_image --epoch 1800 --is_final --gpu-id 2 --clTE --scheduler snap_shot


# dirichlet
python exp_noisy_label_train.py -a resnet --depth 110 --clTE --init_factor 5000 --process_type dirichlet --gpu_id 2 --is_final

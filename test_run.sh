# 测试TCR方法对于无噪声数据集的效果 对比无噪声的结果
python exp_noisy_label_train.py -a resnet --depth 20 --gpu-id 6 --use_TCR
python exp_noisy_label_train.py -a resnet --depth 20 --gpu-id 6 --use_TCR


# 测试不同的时间集成模式对结果的影响
#(1) 损失函数改成KL散度：准确率只有70多？
python exp_noisy_label_train.py -a resnet --depth 20 --gpu-id 6 --clTE --loss_type KLD

#（2）损失函数看成两个部分的加和
python exp_noisy_label_train.py -a resnet --depth 20 --gpu-id 6 --clTE --model_type combination

#（3）尝试以上两种参数的不同模式
python exp_noisy_label_train.py -a resnet --depth 20 --gpu-id 7 --clTE --model_type combination --TE_start_epoch 60
python exp_noisy_label_train.py -a resnet --depth 20 --gpu-id 7 --clTE --model_type combination --R 0.7
python exp_noisy_label_train.py -a resnet --depth 20 --gpu-id 7 --clTE --model_type combination --P 1
python exp_noisy_label_train.py -a resnet --depth 20 --gpu-id 7 --clTE --model_type combination --P 1 --TE_start_epoch 60
p

# (4) 测试use_penalizing_output的效果:也比baseline要低了？检查下代码
python exp_noisy_label_train.py -a resnet --depth 20 --gpu-id 3 --use_penalizing_output
python exp_noisy_label_train.py -a resnet --depth 20 --gpu-id 3 --use_penalizing_output --dataset cifar100
p

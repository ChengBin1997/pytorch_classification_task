cd ..
#python exp_noisy_label_train.py --cfg configer/alexnet.json --use_variance_control --vc_is_count
for beta in {0,0.2,0.5,0.7,1}
do 
	python exp_noisy_label_train.py --cfg configer/alexnet.json --use_variance_control --vc_beta $beta --is_final
done 

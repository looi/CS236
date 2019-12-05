#!/bin/bash
for i in one-hot,300,onehot infersent,4196,infersent bert,868,bert sentence-bert,868,sentence_bert; do 
	IFS=',' read item1 item2 item3 <<< "${i}"
	echo "${item1}" and "${item2}" and "${item3}"
	python train_acgan.py --outf=outputs/acgan_${item3}_64 --conditioning=${item1} --nz=${item2} --model_checkpoint_epoch=2000 --gen_only2 --batch_size=400 --use_cuda=0
done

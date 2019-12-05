#!/bin/bash
for j in {1..5}; do
	for i in one-hot,300,onehot infersent,4196,infersent bert,868,bert sentence-bert,868,sentence_bert; do 
		IFS=',' read item1 item2 item3 <<< "${i}"
		echo "${item1}" and "${item2}" and "${item3}"
		#python train_acgan.py --outf=outputs/acgan_${item3}_64 --conditioning=${item1} --nz=${item2} --model_checkpoint_epoch=2000 --gen_only --force_caption='this is a bird with a grey belly and brown wings.' --force_class=118 --gen_num=$j
		#python train_acgan.py --outf=outputs/acgan_${item3}_64 --conditioning=${item1} --nz=${item2} --model_checkpoint_epoch=2000 --gen_only --force_caption='a bright blue bird with a squat, fat lavender bill, lavender tarsus and feet, and white and gold accents on its wings.' --force_class=54 --gen_num=$j
		#python train_acgan.py --outf=outputs/acgan_${item3}_64 --conditioning=${item1} --nz=${item2} --model_checkpoint_epoch=2000 --gen_only --force_caption='this bird is yellow and brown in color, and has a black beak.' --force_class=173 --gen_num=$j
		python train_acgan.py --outf=outputs/acgan_${item3}_64 --conditioning=${item1} --nz=${item2} --model_checkpoint_epoch=2000 --gen_only --force_caption='this is a bird with a white belly, a blue wing and head and a small black beak.' --force_class=164 --gen_num=$j
		python train_acgan.py --outf=outputs/acgan_${item3}_64 --conditioning=${item1} --nz=${item2} --model_checkpoint_epoch=2000 --gen_only --force_caption='this bird is yellow with black and has a very short beak.' --force_class=47 --gen_num=$j
	done
done

#!/bin/bash

lr=0.01
# lr=0.1
steps=10
max_norm=0.03 
data=cifar10
#data=stl10
#data=mnist
#data=restricted_imagenet
#data=tiny_imagenet
root=data
#root=tiny_imagenet
cp=0.1
model=wide_resnet
#model=aaron
#model=vgg
#model=resnet
#model=resnet34
#model=small_cnn
model_out=./checkpoint/${data}_${model}_adv
echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=3 python3 ./main_adv_cus.py \
                        --lr ${lr} \
                        --data ${data} \
                        --model ${model} \
                        --step ${steps} \
                        --max_norm ${max_norm} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        --corruption_prob ${cp} \
                        --train_sampler true 

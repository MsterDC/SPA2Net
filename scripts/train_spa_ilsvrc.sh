#!/bin/sh

cd ../exper/

python train_cam_spa.py \
    --arch=vgg_spa \
    --epoch=20 \
    --lr=0.001 \
    --batch_size=64 \
    --gpus=0,1,2,3 \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/img_train \
    --train_list=../data/ILSVRC/list/train_list.txt \
    --num_classes=1000 \
    --resume=False \
    --pretrained_model=vgg16.pth \
    --seed=0 \
    --snapshot_dir=../snapshots/vgg_16_ram_s5_.5_bg_.1_gap.2_ilsvrc_20e_10_15d \
    --log_dir=../log/vgg_16_ram_s5_.5_bg_.1_gap_.2_ilsvrc_20e_10_15d \
    --onehot=False \
    --decay_point=10,15 \
    --ram \
    --ram_start=5 \
    --ra_loss_weight=0.5 \
    --ram_th_bg=0.1 \
    --ram_bg_fg_gap=0.2 \


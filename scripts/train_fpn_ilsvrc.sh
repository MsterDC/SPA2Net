#!/bin/sh

cd ../exper/

python train_fpn.py \
    --arch=vgg_fpn \
    --epoch=25 \
    --lr=0.001 \
    --rcst_lr=0.000005 \
    --batch_size=256 \
    --gpus=0,1,2,3 \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/img_train \
    --train_list=../data/ILSVRC/list/train_list.txt \
    --num_classes=1000 \
    --resume=False \
    --pretrained_model=vgg16.pth \
    --seed=0 \
    --snapshot_dir=../snapshots/vgg_16_ram_ilsvrc_rcst_t4 \
    --log_dir=../log/vgg_16_ram_ilsvrc_t4 \
    --onehot=False \
    --decay_points=10,15 \
    --rcst \
    --rcst_start=10 \
    --rcst_loss_weight=0.1 \
    --ram \
    --ram_start=5 \
    --ra_loss_weight=0.5 \
    --ram_th_bg=0.1 \
    --ram_bg_fg_gap=0.2 \


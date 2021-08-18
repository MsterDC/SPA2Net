#!/bin/sh

cd ../exper/

python train_spa_sa.py \
    --arch=vgg_spa_sa \
    --epoch=100 \
    --lr=0.001 \
    --batch_size=128 \
    --gpus=0,1 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --train_list=../data/CUB_200_2011/list/train.txt \
    --num_classes=200 \
    --resume=False \
    --pretrained_model=vgg16.pth \
    --seed=0 \
    --snapshot_dir=../snapshots/vgg16_sa+spa_#rep8 \
    --log_dir=../log/vgg16_sa+spa_#rep8 \
    --onehot=False \
    --decay_point=80 \
    --ram_start=20 \
    --ra_loss_weight=0.5 \
    --ram_th_bg=0.1 \
    --ram_bg_fg_gap=0.2 \
    --in_norm=True \
    --sa_lr=0.001 \
    --sa_head=8 \
    --sa_neu_num=512 \
    --sa_edge_encode=True \
    --sa_start=20 \
    --watch_cam \
    --use_thr_pool=False \
    --th_avgpool=0.6 \
    --thr_pool_start=0 \
    --ram \
    --sa \


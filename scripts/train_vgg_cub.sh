#!/bin/sh

cd ../exper/

python train_sst.py \
    --arch=vgg_sst \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --train_list=../data/CUB_200_2011/list/train.txt \
    --num_classes=200 \
    --scg_com \
    --scg_blocks=4,5 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --ram_th_bg=0.1 \
    --ram_bg_fg_gap=0.2 \
    --ram_start=20 \
    --sos_start=0 \
    --sos_gt_seg=True \
    --sos_seg_method=TC \
    --sos_loss_method=BCE \
    --sos_fg_th=0.2 \
    --sos_bg_th=0.1 \
    --sa_use_edge=True \
    --sa_edge_stage=4,5 \
    --sa_start=20 \
    --sa_head=8 \
    --sa_neu_num=512 \
    --snapshot_dir=../snapshots/vgg16_sos+sa_v3_#36 \
    --log_dir=../log/vgg16_sos+sa_v3_#36 \
    --epoch=150 \
    --decay_points=none \
    --decay_module=bb,cls,sa \
    --batch_size=64 \
    --gpus=0,1,2,3 \
    --lr=0.001 \
    --cls_lr=0.01 \
    --sos_lr=0.00005 \
    --sa_lr=0.005 \
    --spa_loss=True \
    --spa_loss_weight=0.0001 \
    --spa_loss_start=20 \
    --sos_loss_weight=0.5 \
    --ram \
    --ra_loss_weight=1 \
    --mode=sos+sa_v3 \
    --watch_cam \

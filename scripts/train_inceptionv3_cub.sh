#!/bin/sh

cd ../exper/

python train_sst.py \
    --arch=inceptionv3_sst \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --train_list=../data/CUB_200_2011/list/train.txt \
    --num_classes=200 \
    --ram_start=20 \
    --scg_com \
    --scg_blocks=4,5 \
    --sos_gt_seg=True \
    --sos_seg_method=TC \
    --sos_loss_method=BCE \
    --sa_edge_stage=4,5 \
    --sa_neu_num=768 \
    --pretrained_model=inception_v3_google.pth \
    --pretrained_model_dir=../pretrained_models \
    --snapshot_dir=../snapshots/inceptionv3_sos+sa_v3_#19 \
    --log_dir=../log/inceptionv3_sos+sa_v3_#19 \
    --epoch=100 \
    --decay_points=none \
    --decay_module=all \
    --warmup=False \
    --warmup_fun=none \
    --batch_size=64 \
    --gpus=0,1,2,3 \
    --lr=0.001 \
    --cls_lr=0.01 \
    --sos_lr=0.00005 \
    --sa_lr=0.01 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --sos_loss_weight=0.5 \
    --sos_start=0 \
    --sos_fg_th=0.4 \
    --sos_bg_th=0.3 \
    --sa_start=20 \
    --sa_head=8 \
    --sa_use_edge=True \
    --ra_loss_weight=0.5 \
    --ram_th_bg=0.1 \
    --ram_bg_fg_gap=0 \
    --mode=sos+sa_v3 \
    --watch_cam \
    --ram \

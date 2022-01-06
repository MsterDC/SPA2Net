#!/bin/sh

cd ../exper/

python train_sst.py \
    --arch=inceptionv3_sst \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/img_train \
    --train_list=../data/ILSVRC/list/train_list.txt \
    --num_classes=1000 \
    --scg_com \
    --sos_start=0 \
    --scg_blocks=4,5 \
    --sos_gt_seg=True \
    --sos_seg_method=TC \
    --sos_loss_method=BCE \
    --sa_use_edge=True \
    --sa_edge_stage=4,5 \
    --sa_head=8 \
    --sa_neu_num=768 \
    --watch_cam \
    --snapshot_dir=../snapshots/ilsvrc/inceptionv3_sos+sa_v3_#21 \
    --log_dir=../log/ilsvrc/inceptionv3_sos+sa_v3_#21 \
    --load_finetune=True \
    --pretrained_model=ilsvrc_epoch_20.pth.tar \
    --pretrained_model_dir=../snapshots/ilsvrc/inceptionv3_cls \
    --batch_size=64 \
    --gpus=0,1,2,3,4,5,6,7 \
    --epoch=20 \
    --warmup=False \
    --warmup_fun=gra \
    --decay_point=15,17 \
    --decay_module=all\;all \
    --lr=0.001 \
    --cls_lr=0.001 \
    --sa_lr=0.01 \
    --sos_lr=0.00005 \
    --scg_fosc_th=0.1 \
    --scg_sosc_th=0.5 \
    --sa_start=0 \
    --sos_fg_th=0.4 \
    --sos_bg_th=0.3 \
    --sos_loss_weight=0.5 \
    --ram_th_bg=0.1 \
    --ram_bg_fg_gap=0.8 \
    --ram_start=0 \
    --ra_loss_weight=0.1 \
    --mode=sos+sa_v3 \
    --ram \


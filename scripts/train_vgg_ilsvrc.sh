#!/bin/sh

cd ../exper/

python train_sst.py \
    --arch=vgg_sst \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/img_train \
    --train_list=../data/ILSVRC/list/train_list.txt \
    --num_classes=1000 \
    --scg_com \
    --sos_start=0 \
    --sa_head=8 \
    --watch_cam \
    --snapshot_dir=../snapshots/ilsvrc/vgg16_sos+sa_v3_finetune_#4 \
    --log_dir=../log/ilsvrc/vgg16_sos+sa_v3_finetune_#4 \
    --load_finetune=True \
    --finetuned_model_dir=../snapshots/ilsvrc/vgg16_spa_#1 \
    --finetuned_model=ilsvrc_epoch_20.pth.tar \
    --batch_size=64 \
    --gpus=0,1,2,3,4,5,6,7 \
    --epoch=20 \
    --decay_point=15,17 \
    --decay_module=all\;all \
    --lr=0.001 \
    --cls_lr=0.001 \
    --sos_lr=0.00005 \
    --sa_lr=0.001 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --sa_start=0 \
    --sa_use_edge=True \
    --sos_fg_th=0.5 \
    --sos_bg_th=0.3 \
    --sos_loss_weight=0.5 \
    --ram \
    --ram_loss_weight=0.5 \
    --ram_start=0 \
    --ram_th_bg=0.1 \
    --ram_bg_fg_gap=0.2 \
    --spa_loss=True \
    --spa_loss_weight=0.001 \
    --spa_loss_start=0 \
    --mode=sos+sa_v3 \


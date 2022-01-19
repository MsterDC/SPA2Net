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
    --sa_head=8 \
    --watch_cam \
    --snapshot_dir=../snapshots/ilsvrc/inceptionv3_sos+sa_v3_#28 \
    --log_dir=../log/ilsvrc/inceptionv3_sos+sa_v3_#28 \
    --drop_last=False \
    --pin_memory=True \
    --warmup=True \
    --warmup_fun=gra \
    --num_workers=12 \
    --resume=False \
    --restore_from=ilsvrc_epoch_18.pth.tar \
    --load_finetune=True \
    --finetuned_model_dir=../snapshots/ilsvrc/inceptionv3_cls \
    --finetuned_model=ilsvrc_epoch_20.pth.tar \
    --batch_size=64 \
    --gpus=0,1,2,3 \
    --epoch=20 \
    --decay_node=12@14 \
    --decay_module=bakb,cls-h,sAtt@bakb,cls-h,sAtt \
    --decay_scale=0.1,0.1,0.1@0.1,0.1,0.1 \
    --lr=0.001 \
    --cls_lr=0.001 \
    --sa_lr=0.005 \
    --sos_lr=0.00005 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --sa_start=3 \
    --sa_use_edge=True \
    --sos_fg_th=0.2 \
    --sos_bg_th=0.1 \
    --sos_loss_weight=0.5 \
    --ram \
    --ram_loss_weight=0.5 \
    --ram_start=3 \
    --ram_th_bg=0.4 \
    --ram_bg_fg_gap=0.1 \
    --spa_loss=False \
    --mode=sos+sa_v3 \

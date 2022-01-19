#!/bin/sh

cd ../exper/

python train_sst.py \
    --arch=vgg_sst \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --train_list=../data/CUB_200_2011/list/train.txt \
    --num_classes=200 \
    --scg_com \
    --sa_head=8 \
    --sos_start=0 \
    --watch_cam \
    --snapshot_dir=../snapshots/vgg16_sos+sa_v3_rept_#2 \
    --log_dir=../log/vgg16_sos+sa_v3_rept_#2 \
    --drop_last=False \
    --pin_memory=True \
    --num_workers=12 \
    --resume=False \
    --restore_from=cub_epoch_100.pth.tar \
    --load_finetune=False \
    --finetuned_model_dir=../snapshots/vgg16_cls \
    --finetuned_model=cub_epoch_100.pth.tar \
    --epoch=100 \
    --decay_node=dynamic \
    --decay_module=bakb,cls-h,loc-h,sAtt \
    --decay_scale=0.1,0.1,0.1,0.1 \
    --batch_size=64 \
    --gpus=0,1 \
    --lr=0.001 \
    --cls_lr=0.01 \
    --sos_lr=0.0001 \
    --sa_lr=0.005 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --sa_start=20 \
    --sa_use_edge=True \
    --sos_loss_weight=0.5 \
    --sos_fg_th=0.2 \
    --sos_bg_th=0.1 \
    --ram \
    --ram_start=20 \
    --ram_loss_weight=0.5 \
    --ram_th_bg=0.1 \
    --ram_bg_fg_gap=0.2 \
    --mode=sos+sa_v3 \

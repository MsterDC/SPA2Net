#!/bin/sh

cd ../exper/

python train_sst.py \
    --arch=inceptionv3_sst \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --train_list=../data/CUB_200_2011/list/train.txt \
    --num_classes=200 \
    --sa_head=8 \
    --ram_start=20 \
    --scg_com \
    --watch_cam \
    --snapshot_dir=../snapshots/inceptionv3_sos+sa_v3_#19 \
    --log_dir=../log/inceptionv3_sos+sa_v3_#19 \
    --load_finetune=False \
    --finetuned_model_dir=../snapshots/inceptionv3_cls \
    --finetuned_model=cub_epoch_100.pth.tar \
    --epoch=100 \
    --decay_points=none \
    --decay_module=all \
    --batch_size=128 \
    --gpus=0,1,2,3,4,5,6,7 \
    --lr=0.001 \
    --cls_lr=0.01 \
    --sos_lr=0.00005 \
    --sa_lr=0.01 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --sa_start=20 \
    --sa_use_edge=True \
    --sos_start=0 \
    --sos_loss_weight=0.5 \
    --sos_fg_th=0.4 \
    --sos_bg_th=0.3 \
    --ram \
    --ram_loss_weight=0.5 \
    --ram_th_bg=0.1 \
    --ram_bg_fg_gap=0 \
    --mode=sos+sa_v3 \


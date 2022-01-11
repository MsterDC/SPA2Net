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
    --snapshot_dir=../snapshots/ilsvrc/inceptionv3_sos+sa_v3_#18 \
    --log_dir=../log/ilsvrc/inceptionv3_sos+sa_v3_#18 \
    --num_workers=32 \
    --resume=True \
    --restore_from=ilsvrc_epoch_4.pth.tar \
    --load_finetune=False \
    --finetuned_model_dir=../snapshots/ilsvrc/inceptionv3_cls \
    --finetuned_model=ilsvrc_epoch_20.pth.tar \
    --batch_size=64 \
    --gpus=0,1,2,3 \
    --epoch=20 \
    --warmup=False \
    --warmup_fun=gra \
    --decay_point=5,17 \
    --decay_module=bb,cls,sa\;bb,cls,sa \
    --lr=0.001 \
    --cls_lr=0.001 \
    --sa_lr=0.01 \
    --sos_lr=0.0005 \
    --scg_fosc_th=0.1 \
    --scg_sosc_th=0.5 \
    --sa_start=0 \
    --sa_use_edge=True \
    --sos_fg_th=0.5 \
    --sos_bg_th=0.3 \
    --sos_loss_weight=0.5 \
    --ram \
    --ram_th_bg=0.4 \
    --ram_bg_fg_gap=0.1 \
    --ram_start=0 \
    --ram_loss_weight=0.5 \
    --mode=sos+sa_v3 \

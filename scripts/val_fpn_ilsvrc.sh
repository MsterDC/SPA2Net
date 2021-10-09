#!/bin/sh

cd ../exper/

python val_fpn.py \
    --arch=vgg_fpn \
    --gpus=0 \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/img_val \
    --test_list=../data/ILSVRC/list/val_list.txt \
    --test_box=../data/ILSVRC/list/val_bboxes.txt \
    --num_classes=1000 \
    --snapshot_dir=../snapshots/vgg_16_ram_ilsvrc_rcst_t4 \
    --onehot=False \
    --debug_dir=../debug/vgg_16_.2_fo_1_so_.1_fg_.05_bg \
    --restore_from=ilsvrc_epoch_20.pth.tar  \
    --threshold=0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5 \
    --scg \
    --scg_layers=bb \
    --scg_com \
    --scg_blocks=4,5 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --scg_so_weight=1 \
    --scg_fg_th=0.1 \
    --scg_bg_th=0.05 \
    --scg_order=2 \

#!/bin/sh

cd ../exper/

python  val_spa.py \
    --arch=vgg_spa \
    --gpus=1 \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/img_val \
    --test_list=../data/ILSVRC/list/val_list.txt \
    --test_box=../data/ILSVRC/list/val_bboxes.txt \
    --num_classes=1000 \
    --snapshot_dir=../snapshots/vgg_16_ram_s5_.5_bg_.1_gap.2_ilsvrc_20e_10_15d \
    --onehot=False \
    --debug_dir=../debug/baseline_.1_fo_.5_so_.05_fg_.01_bg_scgV1 \
    --restore_from=ilsvrc_epoch_20.pth.tar \
    --threshold=0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5 \
    --scg \
    --scg_com \
    --scg_blocks=4,5 \
    --scg_fosc_th=0.1 \
    --scg_sosc_th=0.5 \
    --scg_so_weight=1 \
    --scg_fg_th=0.05 \
    --scg_bg_th=0.01 \
    --scg_order=2 \
 

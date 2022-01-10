#!/bin/sh

cd ../exper/

python test_sst.py \
    --arch=vgg_sst \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/img_val \
    --test_list=../data/ILSVRC/list/val_list.txt \
    --test_box=../data/ILSVRC/list/val_bboxes.txt \
    --num_classes=1000 \
    --sa_head=8 \
    --scg_com \
    --snapshot_dir=../snapshots/ilsvrc/vgg16_spa+sa_#6 \
    --debug_dir=../debug/ilsvrc/vgg16_spa+sa_#6_t1 \
    --threshold=0.1,0.8 \
    --batch_size=20 \
    --gpus=1 \
    --restore_from=ilsvrc_epoch_20.pth.tar \
    --scg_version=v2 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --sa_use_edge=False \
    --mask_save=False \
    --mask_path=../results \
    --mask_only=False \
    --debug \
    --debug_num=10 \
    --debug_only=False \
    --mode=spa+sa \

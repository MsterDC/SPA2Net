#!/bin/sh

cd ../exper/

python test_sst.py \
    --arch=vgg_sst \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --test_list=../data/CUB_200_2011/list/test.txt \
    --test_box=../data/CUB_200_2011/list/test_boxes.txt \
    --num_classes=200 \
    --scg_com \
    --snapshot_dir=../snapshots/vgg16_sos+sa_v3_rept_#1 \
    --debug_dir=../debug/vgg16_sos+sa_v3_rept_#1_t1 \
    --batch_size=15 \
    --restore_from=cub_epoch_108.pth.tar \
    --threshold=0.05,0.5 \
    --scg_version=v2 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --gpus=0 \
    --sa_head=8 \
    --sa_use_edge=True \
    --debug \
    --debug_num=10 \
    --debug_only=False \
    --mode=sos+sa_v3 \

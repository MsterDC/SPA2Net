#!/bin/sh

cd ../exper/

python test_sst.py \
    --arch=inceptionv3_sst \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --test_list=../data/CUB_200_2011/list/test.txt \
    --test_box=../data/CUB_200_2011/list/test_boxes.txt \
    --num_classes=200 \
    --sa_head=8 \
    --scg_com \
    --snapshot_dir=../snapshots/inceptionv3_sos+sa_v3_#14 \
    --debug_dir=../debug/inceptionv3_sos+sa_v3_#14_test \
    --batch_size=3 \
    --gpus=0 \
    --restore_from=cub_epoch_119.pth.tar \
    --threshold=0.15,0.25 \
    --scg_version=v2 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --sa_use_edge=True \
    --debug \
    --debug_num=10 \
    --debug_only=False \
    --mode=sos+sa_v3 \

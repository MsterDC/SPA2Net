#!/bin/sh

cd ../exper/

python test_sst.py \
    --arch=inceptionv3_sst \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/img_val \
    --test_list=../data/ILSVRC/list/val_list.txt \
    --test_box=../data/ILSVRC/list/val_bboxes.txt \
    --num_classes=1000 \
    --sa_head=8 \
    --scg_com \
    --snapshot_dir=../snapshots/inceptionv3_sos+sa_v3_#14 \
    --debug_dir=../debug/inceptionv3_sos+sa_v3_#14_test \
    --threshold=0.15,0.25 \
    --batch_size=3 \
    --gpus=0 \
    --restore_from=cub_epoch_119.pth.tar \
    --scg_version=v2 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --sa_use_edge=True \
    --mask_save=False \
    --mask_path=../results \
    --mask_only=False \
    --debug \
    --debug_num=10 \
    --debug_only=False \
    --mode=spa+sa \

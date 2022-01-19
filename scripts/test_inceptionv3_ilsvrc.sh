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
    --snapshot_dir=../snapshots/ilsvrc/inceptionv3_sos+sa_v3_#24 \
    --debug_dir=../debug/ilsvrc/inceptionv3_sos+sa_v3_#24_t1 \
    --threshold=0.05,0.8 \
    --batch_size=5 \
    --gpus=0 \
    --restore_from=ilsvrc_epoch_20.pth.tar \
    --scg_version=v2 \
    --scg_fosc_th=0.1 \
    --scg_sosc_th=0.5 \
    --sa_use_edge=True \
    --mask_save=False \
    --mask_path=../results \
    --mask_only=False \
    --debug \
    --debug_num=10 \
    --debug_only=False \
    --mode=sos+sa_v3 \

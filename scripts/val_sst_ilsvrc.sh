#!/bin/sh

cd ../exper/

python val_sst.py \
    --arch=vgg_sst \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/img_val \
    --test_list=../data/ILSVRC/list/val_list.txt \
    --test_box=../data/ILSVRC/list/val_bboxes.txt \
    --num_classes=1000 \
    --onehot=False \
    --in_norm=True \
    --scg_com \
    --scg_blocks=4,5 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --scg_so_weight=1 \
    --scg_order=2 \
    --gpus=0 \
    --snapshot_dir=../snapshots/ilsvrc/vgg16_mc_sos_#1 \
    --debug_dir=../debug/ilsvrc/vgg16_mc_sos_#1_t1 \
    --restore_from=ilsvrc_epoch_20.pth.tar \
    --threshold=0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5 \
    --sos_method=BC \
    --sa_use_edge=True \
    --sa_edge_stage=5 \
    --sa_head=8 \
    --sa_neu_num=512 \
    --use_tap=False \
    --tap_th=0.1 \
    --mode=mc_sos \
    --debug \

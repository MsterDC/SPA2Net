#!/bin/sh

cd ../exper/

python val_sst_quick.py \
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
    --scg_so_weight=1 \
    --scg_order=2 \
    --snapshot_dir=../snapshots/ilsvrc/vgg16_sos+sa_v3_#29 \
    --debug_dir=../debug/ilsvrc/vgg16_sos+sa_v3_#29_t1 \
    --batch_size=20 \
    --restore_from=ilsvrc_epoch_20.pth.tar \
    --scg_version=v2 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --gpus=2 \
    --threshold=0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5 \
    --sos_seg_method=TC \
    --sos_loss_method=BCE \
    --sa_use_edge=True \
    --sa_edge_weight=1 \
    --sa_edge_stage=4,5 \
    --sa_head=1 \
    --sa_neu_num=512 \
    --mode=sos+sa_v3 \
    --debug \

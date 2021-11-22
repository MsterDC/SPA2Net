#!/bin/sh

cd ../exper/

python test_sst.py \
    --arch=vgg_sst \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/img_val \
    --test_list=../data/ILSVRC/list/val_list.txt \
    --test_box=../data/ILSVRC/list/val_bboxes.txt \
    --num_classes=1000 \
    --scg_com \
    --scg_blocks=4,5 \
    --snapshot_dir=../snapshots/ilsvrc/vgg16_sos+sa_v3_wp_#36 \
    --debug_dir=../debug/ilsvrc/vgg16_sos+sa_v3_wp_#36_t2 \
    --batch_size=18 \
    --restore_from=ilsvrc_epoch_15.pth.tar \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --gpus=3 \
    --threshold=0.1,0.8 \
    --sos_seg_method=TC \
    --sos_loss_method=BCE \
    --sa_use_edge=True \
    --sa_edge_stage=4,5 \
    --sa_head=8 \
    --sa_neu_num=512 \
    --mode=sos+sa_v3 \
    --debug \
#    --debug_num=10 \
#    --debug_only \

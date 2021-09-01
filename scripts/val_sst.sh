#!/bin/sh

cd ../exper/

python val_sst.py \
    --arch=vgg_sst \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --test_list=../data/CUB_200_2011/list/test.txt \
    --test_box=../data/CUB_200_2011/list/test_boxes.txt \
    --num_classes=200 \
    --onehot=False \
    --in_norm=True \
    --scg_com \
    --scg_blocks=4,5 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --scg_so_weight=1 \
    --scg_order=2 \
    --restore_from=cub_epoch_100.pth.tar \
    --snapshot_dir=../snapshots/vgg16_sos+sa_v2_#3 \
    --debug_dir=../debug/vgg16_sos+sa_v2_#3_t1 \
    --gpus=0 \
    --threshold=0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5 \
    --sos_seg_method=BC \
    --sos_loss_method=BCE \
    --sa_use_edge=True \
    --sa_edge_stage=4,5 \
    --sa_head=4 \
    --sa_neu_num=512 \
    --use_tap=False \
    --tap_th=0.1 \
    --mode=sos+sa \
    --debug \

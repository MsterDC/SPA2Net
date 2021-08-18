#!/bin/sh

cd ../exper/

python val_spa_sa.py \
    --arch=vgg_spa_sa \
    --gpus=0 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --test_list=../data/CUB_200_2011/list/test.txt \
    --test_box=../data/CUB_200_2011/list/test_boxes.txt \
    --num_classes=200 \
    --snapshot_dir=../snapshots/vgg16_spa+tap_#5 \
    --onehot=False \
    --debug_dir=../debug/vgg16_spa+tap_#5_t1 \
    --restore_from=cub_epoch_100.pth.tar \
    --threshold=0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6 \
    --scg \
    --scg_com \
    --scg_blocks=4,5 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --scg_so_weight=1 \
    --scg_fg_th=0.1 \
    --scg_bg_th=0.05 \
    --scg_order=2 \
    --in_norm=True \
    --sa_edge_encode=True \
    --sa_head=8 \
    --sa_neu_num=512 \
    --use_sa=False \
    --th_avgpool=0.6 \
    --use_thr_pool=True \
    --debug \
#    --sa \

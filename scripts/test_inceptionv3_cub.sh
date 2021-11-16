#!/bin/sh

cd ../exper/

python test_sst.py \
    --arch=inceptionv3_sst \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --test_list=../data/CUB_200_2011/list/test.txt \
    --test_box=../data/CUB_200_2011/list/test_boxes.txt \
    --num_classes=200 \
    --sos_seg_method=TC \
    --sos_loss_method=BCE \
    --sa_use_edge=True \
    --sa_edge_stage=4,5 \
    --snapshot_dir=../snapshots/inceptionv3_spa_#10 \
    --debug_dir=../debug/inceptionv3_spa_#10_tt \
    --batch_size=15 \
    --restore_from=cub_epoch_100.pth.tar \
    --threshold=0.05,0.1,0.15,0.2,0.25 \
    --scg_version=v1 \
    --scg_com \
    --scg_blocks=4,5 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=0.5 \
    --gpus=0 \
    --sa_head=8 \
    --sa_neu_num=768 \
    --mode=spa \
    --debug \

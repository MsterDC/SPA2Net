#!/bin/sh

cd ../exper/

python test_sst.py \
    --arch=inceptionv3_sst \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/img_val \
    --test_list=../data/ILSVRC/list/val_list.txt \
    --test_box=../data/ILSVRC/list/val_bboxes.txt \
    --num_classes=1000 \
    --scg_com \
    --scg_blocks=4,5 \
    --sos_seg_method=TC \
    --sos_loss_method=BCE \
    --sa_use_edge=True \
    --sa_edge_stage=4,5 \
    --sa_head=8 \
    --sa_neu_num=768 \
    --snapshot_dir=../snapshots/ilsvrc/inceptionv3_spa_#5 \
    --debug_dir=../debug/ilsvrc/inceptionv3_spa_#5_t1 \
    --gpus=0 \
    --threshold=0.1,0.5 \
    --batch_size=10 \
    --restore_from=ilsvrc_epoch_20.pth.tar \
    --scg_version=v1 \
    --scgv1_bg_th=0.05 \
    --scgv1_fg_th=0.05 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=0.5 \
    --mode=spa \
    --debug \

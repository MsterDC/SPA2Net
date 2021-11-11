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
    --snapshot_dir=../snapshots/ilsvrc/inceptionv3_spa_#2 \
    --debug_dir=../debug/ilsvrc/inceptionv3_spa_#2_t1 \
    --batch_size=20 \
    --restore_from=ilsvrc_epoch_20.pth.tar \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --gpus=0 \
    --threshold=0.6,0.65,0.7,0.75,0.8 \
    --sos_seg_method=TC \
    --sos_loss_method=BCE \
    --sa_use_edge=True \
    --sa_edge_stage=4,5 \
    --sa_head=8 \
    --sa_neu_num=768 \
    --mode=sos+sa_v3 \
    --debug \

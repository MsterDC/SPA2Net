#!/bin/sh

arch="inceptionv3_sst"
restore_from="cub_epoch_100.pth.tar"
snapshot_dir="../../snapshots/inceptionv3_cls_#1"
save_dir="../../evalbox/inceptionv3_cls_#1"
gpus="1"
batch_size=15
threshold="0.15,0.15"
scg_fosc_th=0.2
scg_sosc_th=1
sa_head=8
mode="spa"
vis_bbox_num=1000

python eval_bbox.py \
    --arch=${arch} \
    --dataset=cub \
    --img_dir=../../data/CUB_200_2011/images \
    --test_list=../../data/CUB_200_2011/list/test.txt \
    --test_box=../../data/CUB_200_2011/list/test_boxes.txt \
    --num_classes=200 \
    --scg_com \
    --scg_blocks=4,5 \
    --sos_seg_method=TC \
    --sos_loss_method=BCE \
    --sa_edge_stage=4,5 \
    --sa_use_edge=True \
    --sa_head=${sa_head} \
    --sa_neu_num=${sa_neu_num} \
    --snapshot_dir=${snapshot_dir} \
    --save_dir=${save_dir} \
    --batch_size=${batch_size} \
    --restore_from=${restore_from} \
    --threshold=${threshold} \
    --scg_fosc_th=${scg_fosc_th} \
    --scg_sosc_th=${scg_sosc_th} \
    --gpus=${gpus} \
    --mode=${mode} \
    --vis_bbox \
    --vis_bbox_num=${vis_bbox_num} \
#    --vis_attention \
#    --statis_bbox \

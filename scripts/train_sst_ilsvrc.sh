#!/bin/sh

cd ../exper/

python train_sst.py \
    --arch=vgg_sst \
    --epoch=20 \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/img_train \
    --train_list=../data/ILSVRC/list/train_list.txt \
    --num_classes=1000 \
    --resume=False \
    --pretrained_model=vgg16.pth \
    --seed=0 \
    --onehot=False \
    --decay_point=10,15 \
    --in_norm=True \
    --ram_start=5 \
    --ra_loss_weight=0.5 \
    --ram_th_bg=0.1 \
    --ram_bg_fg_gap=0.2 \
    --scg_com \
    --scg_blocks=4,5 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --scg_so_weight=1 \
    --scg_order=2 \
    --use_tap=False \
    --tap_th=0.1 \
    --tap_start=0 \
    --snapshot_dir=../snapshots/ilsvrc/vgg16_mc_sos_#1 \
    --log_dir=../log/ilsvrc/vgg16_mc_sos_#1 \
    --batch_size=64 \
    --gpus=0,1,2,3 \
    --lr=0.001 \
    --sos_lr=0.01 \
    --sos_gt_seg=True \
    --sos_seg_method=BC \
    --sos_fg_th=0.2 \
    --sos_bg_th=0.2 \
    --sos_loss_weight=0.5 \
    --sos_start=0 \
    --sa_lr=0.001 \
    --sa_use_edge=True \
    --sa_edge_stage=5 \
    --sa_start=20 \
    --sa_head=8 \
    --sa_neu_num=512 \
    --watch_cam \
    --hinge_lr=0.00005 \
    --hinge_loss_weight=1 \
    --mode=mc_sos \
    --ram \

#    --rcst_lr=0.000005 \
#    --rcst_signal=ori \
#    --rcst_loss_weight=0.1 \
#    --rcst_start=10 \
#    --rcst_ratio=700 \
# mode = spa / sos / spa+sa / sos+sa / spa+hinge
# sos_gt_method = BCE / MSE_BCE / CE2D
# sos_seg = cam / scm / none
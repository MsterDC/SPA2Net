#!/bin/sh

cd ../exper/

python train_sst.py \
    --arch=vgg_sst \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/img_train \
    --train_list=../data/ILSVRC/list/train_list.txt \
    --num_classes=1000 \
    --seed=0 \
    --onehot=False \
    --in_norm=True \
    --use_tap=False \
    --tap_th=0.1 \
    --tap_start=0 \
    --cls_or_hinge=cls \
    --hinge_norm=norm \
    --hinge_p=1 \
    --hinge_m=1 \
    --hinge_lr=0.00005 \
    --hinge_loss_weight=1 \
    --scg_com \
    --scg_so_weight=1 \
    --scg_order=2 \
    --scg_blocks=4,5 \
    --snapshot_dir=../snapshots/ilsvrc/vgg16_sos+sa_v3_wp_#14 \
    --log_dir=../log/ilsvrc/vgg16_sos+sa_v3_wp_#14 \
    --lr=0 \
    --warmup=True \
    --epoch=15 \
    --decay_point=11,13 \
    --weight_decay=0.0005 \
    --batch_size=64 \
    --gpus=0,1,2,3 \
    --load_finetune=True \
    --pretrained_model=ilsvrc_epoch_20.pth.tar \
    --pretrained_model_dir=../snapshots/ilsvrc/vgg16_spa_#1 \
    --resume=False \
    --restore_from=none \
    --ra_loss_weight=1 \
    --ram_start=3 \
    --ram_th_bg=0.1 \
    --ram_bg_fg_gap=0.2 \
    --ram \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=1 \
    --sos_lr=0.00005 \
    --sos_gt_seg=True \
    --sos_seg_method=TC \
    --sos_loss_method=BCE \
    --sos_fg_th=0.2 \
    --sos_bg_th=0.1 \
    --sos_loss_weight=0.5 \
    --sos_start=0 \
    --sa_lr=0.001 \
    --sa_use_edge=True \
    --sa_edge_stage=4,5 \
    --sa_edge_weight=1 \
    --sa_start=3 \
    --sa_head=8 \
    --sa_neu_num=512 \
    --watch_cam \
    --mode=sos+sa_v3 \


# mode = spa / sos / spa+sa / sos+sa_v1 / sos+sa_v2 / sos+sa_v3
# sos_loss_method = BCE / MSE
# sos_seg_method = TC / BC
#    --rcst_lr=0.000005 \
#    --rcst_signal=ori \
#    --rcst_loss_weight=0.1 \
#    --rcst_start=10 \
#    --rcst_ratio=700 \
#!/bin/sh

cd ../exper/

python train_sst.py \
    --arch=vgg_sst \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --train_list=../data/CUB_200_2011/list/train.txt \
    --num_classes=200 \
    --seed=0 \
    --onehot=False \
    --in_norm=True \
    --scg_com \
    --scg_blocks=4,5 \
    --scg_so_weight=1 \
    --scg_order=2 \
    --use_tap=False \
    --tap_th=0.1 \
    --tap_start=0 \
    --cls_or_hinge=cls \
    --hinge_norm=norm \
    --hinge_p=1 \
    --hinge_m=1 \
    --hinge_lr=0.00005 \
    --hinge_loss_weight=1 \
    --snapshot_dir=../snapshots/vgg16_sos+sa_v3_#28 \
    --log_dir=../log/vgg16_sos+sa_v3_#28 \
    --epoch=150 \
    --decay_points=none \
    --load_finetune=False \
    --pretrained_model=vgg16.pth \
    --pretrained_model_dir=../pretrained_models \
    --resume=False \
    --restore_from=none \
    --batch_size=64 \
    --gpus=0,1 \
    --lr=0.001 \
    --weight_decay=0.0005 \
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
    --sa_lr=0.005 \
    --sa_use_edge=True \
    --sa_edge_weight=1 \
    --sa_edge_stage=4,5 \
    --sa_start=20 \
    --sa_head=8 \
    --sa_neu_num=512 \
    --watch_cam \
    --warmup=False \
    --mode=sos+sa_v3 \
    --ram_th_bg=0.1 \
    --ram_bg_fg_gap=0.2 \
    --ram_start=20 \
    --ra_loss_weight=1 \
    --ram \


#    --rcst_lr=0.000005 \
#    --rcst_signal=ori \
#    --rcst_loss_weight=0.1 \
#    --rcst_start=10 \
#    --rcst_ratio=700 \
# mode = spa / sos / mc_sos / mc_sos+sa / spa+sa / sos+sa / spa+hinge
# sos_seg_method = TC / BC
#!/bin/sh

backbone="vgg16"
arch="sos+sa_v3_wp"
exp_num="#1"
exp_id="${backbone}_${arch}_${exp_num}"
root_dir="../../results"

python eval_mask.py \
    --root_dir=${root_dir} \
    --exp_id=${exp_id} \
    --dataset=ilsvrc \
    --cam_curve_interval=0.05 \
    --eval_cam=1 \
    --eval_scg=1 \
    --eval_sos=1 \
    --save_pr_curve=1 \
    --save_iouth_curve=1 \

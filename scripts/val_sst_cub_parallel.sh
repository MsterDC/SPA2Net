#!/bin/sh
cd ../exper/

function PushQue {
    Que="$Que $1"
    Nrun=$(($Nrun+1))
}

function GenQue {
    OldQue=$Que
    Que="";Nrun=0
    for PID in $OldQue; do
      if [[ -d /proc/$PID ]]; then
        PushQue $PID
      fi
    done
}

function ChkQue {
    OldQue=$Que
    for PID in $OldQue;do
      if [[ ! -d /proc/$PID ]]; then
      GenQue;break
      fi
    done
}

list_prefix="../data/CUB_200_2011/list/split/test_"
box_prefix="../data/CUB_200_2011/list/split/test_boxes_"
debug_prefix="../debug/"
debug_suffix="_t1/id_t1-"
snap_prefix="../snapshots/"
task_name="vgg16_sos+sa_v3_#1"
file_suffix=".txt"
gpu_num=4
snapshots=$snap_prefix$task_name

Njob=19
Nproc=20

for ((i = 0; i<=$Njob; i++)); do
    test_list=$list_prefix$i$file_suffix
    box_list=$box_prefix$i$file_suffix
    debug=$debug_prefix$task_name$debug_suffix$i
    gpuid=$(( $i%$gpu_num ))
    echo ${test_list}
    echo ${box_list}
    echo ${snapshots}
    echo ${debug}
    python val_sst.py \
        --arch=vgg_sst \
        --dataset=cub \
        --img_dir=../data/CUB_200_2011/images \
        --test_list=$test_list \
        --test_box=$box_list \
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
        --snapshot_dir=$snapshots \
        --debug_dir=$debug \
        --scg_version=v2 \
        --gpus=$gpuid \
        --threshold=0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5 \
        --sos_seg_method=TC \
        --sos_loss_method=BCE \
        --sa_use_edge=True \
        --sa_edge_stage=4,5 \
        --sa_head=4 \
        --sa_neu_num=512 \
        --use_tap=False \
        --tap_th=0.1 \
        --mode=sos \
        --debug &
    PID=$!
    PushQue $PID
    while [[ $Nrun -ge $Nproc ]]; do
        ChkQue
    done
done
echo -e "time-consuming: $SECONDS seconds"
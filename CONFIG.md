# Parameter Configuration

This document contains detailed configuration for parameters using in training and testing.

## :deciduous_tree: Train
* --snapshot_dir: prefixes: `../snapshots/ilsvrc/` for `ILSVRC` and `../snapshots/` for `CUB`, and the following format is recommended: `Base_Mode_#ExpNumber`, such as `../snapshots/ilsvrc/vgg16_sos+sa_v3_#1`.
* --log_dir: prefixes: `../log/ilsvrc/` for `ILSVRC` and `../log/` for `CUB`, and the following dir name is the same as `snapshot_dir`, such as `../log/ilsvrc/vgg16_sos+sa_v3_#1`.
* --lr: learning rate setting for **backbone** and **classification-head**, default `0.001` (cls_head will be ×10 when training, i.e. equals to `0.01`).
* --warmup: using warmup strategy or not, 'True' or 'False'. (See the end for more details)
* --weight_decay: SGD optimizer used, default is `0.0005`.
* --load_finetune: 'True' or 'False' for loading fine-tuned classification model(includes cls_head) on ILSVRC. (To solve the problem of square activation)
* --pretrained_model: fine-tuned model name, default is 'ilsvrc_epoch_20.pth.tar'
* --pretrained_model_dir: fine_tuned model path, default is '../snapshots/ilsvrc/vgg16_spa_#1'
* --resume: resume training from checkpoint or not, 'False' or 'True', default is '`False`'.
* --restore_from: default is '`none`', or need to specify the path of checkpoint.
* --ram_th_bg : when using `SCGv2` in SPA on ILSVRC, `0.1` is better, but `0.4` is **better** for `SCGv1` on ILSVRC. 
* --ram_bg_fg_gap: when using `SCGv2` in SPA on ILSVRC, `0.2` is better, but `0.1` is **better** for `SCGv1` on ILSVRC. 
* --scg_fosc_th: when using `SCGv2` in SPA on ILSVRC, `0.2` is better, but `0.1` is **default setting** in paper for `SCGv1` on ILSVRC. 
* --scg_sosc_th: when using `SCGv2` in SPA on ILSVRC, `1` is better, but `0.5` is **default setting** in paper for `SCGv1` on ILSVRC. 
* --sos_lr: learning rate of sos_head, defalut is `5e-0.5`.
* --sos_gt_seg: using segmentation or not when generate pasedo ground-truth for predicting `sos_map`. 'True' or 'False', default is '`True`'.
* --sos_seg_method: when setting `sos_gt_seg=True` , sos_gt_seg specify the segmetation method, 
'TC' and 'BC' is supported for double-thresholding and single-thresholding, respectively. Default is '`TC`'.
* --sos_loss_method: loss function for SOS branch, BCE loss function and MSE loss function are supported, default is '`BCE`'.
* --sos_fg_th: foreground segmentation threshold, default is `0.2` on CUB, it is not clear the value on ILSVRC.
* --sos_bg_th: background segmentation threshold, default is `0.1` on CUB, it is not clear the value on ILSVRC. (Note: when `sos_seg_method` is set to '`BC`', it is invalid)
* --sos_start: SOS branch start training epoch, default is `0` on CUB and ILSVRC.
* --sa_lr: learning rate of self-attention module, defalut is `0.005` on CUB, `0.001` on ILSVRC.
* --sa_use_edge: using edge-encoding or not, 'True' or 'False', default is '`True`'.
* --sa_edge_weight: weight of edge-encoding when add with attention weight. Default is `1`.
* --sa_edge_stage: specifing the features from certain stages to calculate edge-encoding. Default is '`4,5`'.
* --sa_start: epoch to add SA module for training, default is `20` on CUB, `3` on ILSVRC.
* --sa_head: the number of multi-heads of SA module, default is `8`.
* --sa_neu_num: channel number of SA module. Default is `512` in `VGG16`.
* --watch_cam: save CAM, sos_map and gt_sos_map during training. The specific visualized image is set in exper/train_sst.py, and the default setting is '`True`'.
* --mode: current code support modes include: 
```shell
'spa' / 'sos' / 'spa+sa' / 'sos+sa_v1' / 'sos+sa_v2' / 'sos+sa_v3' (defalut)
```

### :books: Explanation of Warmup in the code
If you use warmup strategy, you need also to set it on `lines 56~69` of `./exper/train_sst.py` before each training.
Among them, `wp_period` and `wp_node` respectively specify the period and training node of each warm-up phase, 
and `wp_ps` specifies the parameters that need to be executed in each warm-up phase.

For example, when 
wp_period = [2,2,2], wp_node = [0,3,6], wp_ps = [['sos_weight','sos_bias'], ['sa_weight','sa_bias'], op_params_list], 
that means the learning rate warmup will be performed on the two parameters'sos_weight' and 'sos_bias' in the 0th to 2nd epoch, 
and the learning rate warmup will be performed on the two parameters of 'sa_weight' and'sa_bias' in the 3rd to 5th epoch.
`op_params_list` represents all parameters in the model, the learning rate will be warmup in the 6th to 8th epoch in this example.
In addition, when warmup is performed on a certain set of parameters (such as'sos_weight' and'sos_bias'), 
the learning rate of **all other parameters will be set to 0** to limit the updating of other modules in training .

All parameter field '`op_params_list`' of the model `sos+sa` include: 
```bash
'sos_weight' / 'sos_bias' / 'sa_weight' / 'sa_bias' / 'other_weight' / 'other_bias' / 'cls_weight' / 'cls_bias'
```
where 'other' refers to the parameters of the `backbone` (such as vgg16) ,'cls' refers to the parameters of the `cls_head`, 
'sos' refers to the parameters of `sos_head`, and 'sa' refers to the parameters of the `self-attention` module.

## :deciduous_tree: Test
* --debug_dir: a suffix is usually added to the test directory: `_#t1` to distinguish different test results of the same model, such as `../debug/ilsvrc/vgg16_sos+sa_v3_wp_#3_t1` ,  `../debug/ilsvrc/vgg16_sos+sa_v3_wp_#3_t2` , etc.
* --scg_version: specify the version of SCG algorithm, '`v2`' is the default setting.
* --sos_seg_method: need to align with the settings during training, default is '`TC`'.
* --sos_loss_method: need to align with the settings during training, default is '`BCE`'.
* --sa_use_edge: using edge-encoding or not, default is '`True`'.
* --sa_edge_weight: need to align with the settings during training, default is `1`.
* --sa_edge_stage: need to align with the settings during training, default is '`4,5`'.
* --sa_head: need to align with the settings during training, default is `8`.
* --sa_neu_num: need to align with the settings during training, default is `512` for VGG16.
* --mode: need to align with the settings during training, default is `sos+sa_v3`.
* --debug: it is not support now, it will be fixed in the future.

# WGOL-TPAMI
PyTorch implementation for Self-Supervised Attention (SSA).

## :gear: Quick Start: Train & Test
#### Switch to the `scripts` directory and execute the following shell code:

:fire: For training
```shell
bash ./train_sst_cub.sh  # For CUB-200-2011
```
```shell
bash ./train_sst_ilsvrc.sh  # For ILSVRC
```

:fire: For testing 
```shell
bash ./val_sst_cub.sh  # Serial version for CUB
```
```shell
bash ./val_sst_cub_parallel.sh  # Multi-threaded serial test version for CUB
```
```shell
bash ./val_sst_cub_quick.sh  # Quick Version for CUB (**Recommend**)
```
```shell
bash ./val_sst_ilsvrc.sh  # Serial version for ILSVRC
```
```shell
bash ./val_sst_ilsvrc_quick.sh  # Quick Version for ILSVRC (**Recommend**)
```


### :pushpin: [Tips]
* There is not dataset directory so you need to create after executing `git clone`.
* Please use the parameter `mode` (Type: String) to control the method you want to train or test. 
* The currently supported `mode` include: `spa` / `spa+sa` / `sos` / `sos+sa_v1` / `sos+sa_v2` / `sos+sa_v3`(Default).
* When you assign parameter `mode`, don't worry about other parameters that are not used in the current `mode`, because the irrelevant parameters will be blocked in the code. 


### :pushpin: [Other]
#### Don't forget to set other parameters related to the current `mode`, especially the settings of the following parameters:

:wrench: For any **training** script in dict `scripts`:

* Note the `line 27` of the two training scripts `train_sst_cub.sh` & `train_sst_ilsvrc.sh`.
Before executing the script `each time`, you may need to confirm `all` the parameters `after line 27`. 
* The parameters `before line 27` are the default settings, unless necessary, you **don't need** to change them.


:wrench: For any **testing** script in dict `scripts`:

* The script `val_sst_*_quick.sh` is used by `default` to test the model.
* Take `val_sst_ilsvrc_quick.sh` as an example, you need to pay attention to the code on `line 20` and **after**. 
You need to double-check each time when you run it to ensure that all parameters are correctly configured.
* The parameters **before** the `20th line` are the default settings, you do not need to change them.

#### :mag: For a more detailed explanation of each parameter, please refer to the [`CONFIG.md`](CONFIG.md) in the scripts directory.

***

## :art: [All Archictures]
### :heavy_check_mark: SOS+SA v3 is the final version (At the end).
* SPA (Baseline)
<img src="https://github.com/KevinDongDong/WGOL-TPAMI/blob/main/images/SPA%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.png" width="633" >

* SPA+SA
<img src="https://github.com/KevinDongDong/WGOL-TPAMI/blob/main/images/SPA%2BSA%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.png" width="633" >

* SPA+Hinge
<img src="https://github.com/KevinDongDong/WGOL-TPAMI/blob/main/images/SPA%2BHinge%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.png" width="633" >

* SOS
<img src="https://github.com/KevinDongDong/WGOL-TPAMI/blob/main/images/SOS%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.png" width="633" >

* Multi-channel SOS
<img src="https://github.com/KevinDongDong/WGOL-TPAMI/blob/main/images/Multi-Channel%20SOS%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.png" width="633" >

* SOS+SA v1
<img src="https://github.com/KevinDongDong/WGOL-TPAMI/blob/main/images/SOS%2BSA%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84_1.png" width="633" >

* SOS+SA v2
<img src="https://github.com/KevinDongDong/WGOL-TPAMI/blob/main/images/SOS%2BSA%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84_2.png" width="633" >

* :heavy_check_mark: SOS+SA v3 (Final Version)
<img src="https://github.com/KevinDongDong/WGOL-TPAMI/blob/main/images/SOS%2BSA%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84_3.png" width="633" >



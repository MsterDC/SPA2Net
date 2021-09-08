# WGOL-TPAMI
PyTorch implementation for SSA.

## How to train or test with the code?
#### Switch to the 'scripts' directory and execute:
```shell
bash ./train_sst.sh
```
```shell
bash ./val_sst.sh
```



### [Tips]
* There is not dataset directory so you need to copy 'data' to current project after executing 'git clone' at first time.
* Please use the parameter "mode" to control the method you want to train or test. ('mode' can be: spa / spa+sa / sos / mc_sos / sos+sa_v2 / sos+sa_v3 / mc_sos+sa_v2 / mc_sos+sa_v3 / spa+hinge)
* When you assign a "mode", don't worry about other parameters that are not used in the current task, they will be ignored when the code is executed. So you don't need to delete them.
*  Don’t forget to re-assign your current task-related parameters, such as 'sos_start' or 'sa_start', etc.


### [Other]
* To execute 'spa' or 'spa+sa' with older code:

```shell
bash ./scripts/train_spa.sh
```
```shell
bash ./scripts/val_spa.sh
```

* You also need to pay attention to the parameters in the script which you want to asign.

***

## [Network Architectures]

* SPA
<img src="https://github.com/KevinDongDong/WGOL-TPAMI/blob/main/SPA%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.png" width="633" >

* SPA+Hinge
<img src="https://github.com/KevinDongDong/WGOL-TPAMI/blob/main/SPA%2BHinge%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.png" width="633" >

* SPA+SA
<img src="https://github.com/KevinDongDong/WGOL-TPAMI/blob/main/SPA%2BSA%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.png" width="633" >

* SOS
<img src="https://github.com/KevinDongDong/WGOL-TPAMI/blob/main/SOS%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.png" width="633" >

* Multi-channel SOS
<img src="https://github.com/KevinDongDong/WGOL-TPAMI/blob/main/Multi-Channel%20SOS%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.png" width="633" >

* SOS+SA v2
<img src="https://github.com/KevinDongDong/WGOL-TPAMI/blob/main/SOS%2BSA%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84_2.png" width="633" >

* SOS+SA v3
<img src="https://github.com/KevinDongDong/WGOL-TPAMI/blob/main/SOS%2BSA%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84_3.png" width="633" >



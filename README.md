# WGOL-TPAMI
## how to train or test with the code
#### bash ./scripts/train_sst.sh
#### bash ./scripts/val_sst.sh
#### - Please use the parameter "mode" to control the method you want to train or test. ('mode' can be: spa / spa+sa / sos / sos+sa)
#### - When you assign a "mode", don't worry about other parameters that are not used in the current task, they will be ignored when the code is executed. So you don't need to delete them.
#### - Don’t forget to re-assign your current task-related parameters, such as 'sos_start' or 'sa_start', etc.


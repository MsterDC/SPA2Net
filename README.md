# WGOL-TPAMI
## How to train or test with the code?
#### switch to the'scripts' directory and execute:
#### bash ./train_sst.sh    ~// for training
#### bash ./val_sst.sh    ~// for testing
#### - There is not dataset directory so you need to copy 'data' to current project after executing 'git clone' at first time.
#### - Please use the parameter "mode" to control the method you want to train or test. ('mode' can be: spa / spa+sa / sos / sos+sa)
#### - When you assign a "mode", don't worry about other parameters that are not used in the current task, they will be ignored when the code is executed. So you don't need to delete them.
#### - Don’t forget to re-assign your current task-related parameters, such as 'sos_start' or 'sa_start', etc.
#### bash ./scripts/train_spa.sh

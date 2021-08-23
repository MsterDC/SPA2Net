# WGOL-TPAMI
## [How to train or test with the code?]
#### Switch to the 'scripts' directory and execute:
#### - bash ./train_sst.sh    // for training
#### - bash ./val_sst.sh    // for testing
### [Tips]
#### - There is not dataset directory so you need to copy 'data' to current project after executing 'git clone' at first time.
#### - Please use the parameter "mode" to control the method you want to train or test. ('mode' can be: spa / spa+sa / sos / sos+sa / spa+hinge)
#### - When you assign a "mode", don't worry about other parameters that are not used in the current task, they will be ignored when the code is executed. So you don't need to delete them.
#### - Don’t forget to re-assign your current task-related parameters, such as 'sos_start' or 'sa_start', etc.
### [Other]
#### To execute 'spa' or 'spa+sa' with older code:
#### - bash ./scripts/train_spa.sh
#### - bash ./scripts/val_spa.sh
#### - You also need to pay attention to the parameters in the script which you want to asign.

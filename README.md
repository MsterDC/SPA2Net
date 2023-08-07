# SPA2Net
PyTorch implementation for SPA2Net.

## :gear: Quick Start: Train & Test

### 🔥 Download fine-tuned model for training Inception-V3 on `ILSVRC`.
* [GoogleDrive](https://drive.google.com/file/d/1JsiRQHV39Cr6DTP1MwjAZ-hG4dztdIQS/view?usp=sharing)
* After downloading, ensure `inceptionv3_cls.zip` in `snapshots/ilsvrc/` and execute following:
```
unzip inceptionv3_cls.zip
```
* The model file will be unzipped in this directory `snapshots/ilsvrc/inceptionv3_cls/`

### 🔥 Download fine-tuned model for training VGG16 on `ILSVRC`.
* [BaiduNetdisk](https://pan.baidu.com/s/1-bsNxmqaheHW72umm408uA) code：6ih2 
* [GoogleDrive](https://drive.google.com/file/d/1BXrgBA09eGZ3UvtFJYYm3FfPDScKA5-g/view?usp=sharing)
* After downloading, put the model into `snapshots/ilsvrc/vgg16_spa_#1/`


### Switch to the `scripts` directory and execute the following shell code:

#### :fire: Train
```shell
bash ./train_vgg_cub.sh  # For VGG-16 trained on CUB-200-2011
bash ./train_inceptionv3_cub.sh  # For Inception V3 trained on CUB-200-2011
```
```shell
bash ./train_vgg_ilsvrc.sh  # For VGG-16 trained on ILSVRC
bash ./train_inceptionv3_ilsvrc.sh  # For Inception V3 trained on ILSVRC
```

#### :fire: Test 
Download the model from [GoogleDrive](https://drive.google.com/drive/folders/1nnO1KNxKL3uq36TopobWI3pyzK4xWTXz?usp=sharing) and put it into `snapshot/vgg16_cub_spa2net/`

Execute the following command:
```shell
bash ./test_vgg_cub.sh  # For VGG-16 test on CUB-200-2011
bash ./test_inceptionv3_cub.sh  # For Inception V3 test on CUB-200-2011
```
```shell
bash ./test_vgg_ilsvrc.sh  # For VGG-16 test on ILSVRC
bash ./test_inceptionv3_ilsvrc.sh  # For Inception V3 test on ILSVRC
```

### :mag: For a more detailed explanation of each parameter, please refer to the [`HELP.md`](HELP.md).


# SPA2Net
PyTorch implementation for SPA2Net.

## :gear: Quick Start: Train & Test

### Switch to the `scripts` directory and execute the following shell code:

#### :fire: Train
```shell
bash ./train_vgg_cub.sh  # For VGG-16 trained on CUB-200-2011
```

#### :fire: Test 
Download the model from [GoogleDrive](https://drive.google.com/drive/folders/1nnO1KNxKL3uq36TopobWI3pyzK4xWTXz?usp=sharing) and put it into `snapshot/vgg16_cub_spa2net/`

Execute the following command:
```shell
bash ./test_vgg_cub.sh
```

### :mag: For a more detailed explanation of each parameter, please refer to the [`HELP.md`](HELP.md).


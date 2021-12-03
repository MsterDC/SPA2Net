# Mask Evaluation
### 基本说明
1. 数据：LID Challenge官方仅提供了ImageNet-1K测试集中的部分数据，
即对应SEM论文的实验部分提到的`val`集合，一共有`23151`个经过标注的gt-mask，目前暂未获得SEM论文中的`test`集合数据；
2. 代码：使用的是经过修改的评测代码，目前仅支持评测Mask指标，可以计算获得
`AP score、iou, precision, recall` 四种指标，可以根据结果绘制`PR曲线`、`IoU-Threshold`曲线；
3. 运行本程序需要修改eval.sh中的参数，以及eval_mask.py的部分参数，
但在运行之前请确保在相应路径下存在待评测方法生成的激活图文件，文件命名规则为:`{img_id}.npy`；

### 注意事项
* LID Challenge 提供数据的 [地址](https://lidchallenge.github.io/challenge.html) 。 
数据存在两处错误： ID尾号为`17615`、`18903`的两个样本的宽和高是相反的，需要人工校正。

* 数据集中`未提供`关于图像原始尺寸的文本文件（代码中为`val_sizes.txt`），
这可以使用`bak`目录下的 `gen_sizes_file.py` 脚本生成。

* 当前代码可以评测`CAM`、`SCG`、`SOS`三种激活图，可根据需要进行修改。

* 本代码同时支持测试`gt-known`，但需要对相应路径等参数进行设置。


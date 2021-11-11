from torch.utils.data import DataLoader
import torch
import numpy as np
import random
import os

from .engine_dataset import dataset as DataSet


def data_loader(args, train=True):

    GLOBAL_WORKER_ID = None

    def _init_fn(worker_id):
        global GLOBAL_WORKER_ID
        GLOBAL_WORKER_ID = worker_id
        # python的字符串hash算法并不是直接遍历字符串每个字符去计算hash，
        # 而是会有一个secret prefix和一个secret suffix，可以认为相当于是给字符串加盐后做hash，
        # 可以规避一些规律输入的情况显然这个secret前后缀的值会直接影响计算结果，
        # 而且它有一个启动时随机生成的机制，只不过，在2.x版本中，这个机制默认是关闭的，
        # 前后缀每次启动都设置为0，除非你改了相关环境变量来要求随机，而在3.x中修改了默认行为，
        # 如果你不配置环境变量，则默认是随机一个前后缀值，这样每次启动都会不同这个环境变量是PYTHONHASHSEED，
        # 无论在2.x还是3.x中，配置为一个正整数，将作为随机种子；配置为0，则secret前后缀默认清零
        # （和2.x默认行为就一样了），配置为空串或“random”，则表示让进程随机生成（和3.x默认行为一样）
        os.environ['PYTHONHASHSEED'] = str(args.seed + worker_id)
        random.seed(10 + worker_id)
        np.random.seed(10 + worker_id)
        torch.manual_seed(10 + worker_id)
        torch.cuda.manual_seed(10 + worker_id)
        torch.cuda.manual_seed_all(10 + worker_id)

    # training and test dataset & dataloader
    if train:
        img_train = DataSet(args, train_flag=train)
        train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  worker_init_fn=_init_fn)
        return train_loader
    else:
        img_test = DataSet(args, train_flag=train)
        test_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        return test_loader

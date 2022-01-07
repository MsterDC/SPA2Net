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
                                  worker_init_fn=_init_fn, drop_last=True, pin_memory=True)
        return train_loader
    else:
        img_test = DataSet(args, train_flag=train)
        test_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        return test_loader

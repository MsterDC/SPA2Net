import os
import torch
import sys

__all__ = ['restore']


def restore(args, model, optimizer, istrain=True, including_opt=False):
    if args.restore_from != '' and ('.pth' in args.restore_from):
        snapshot = os.path.join(args.snapshot_dir, args.restore_from)
    else:
        restore_dir = args.snapshot_dir
        filelist = os.listdir(restore_dir)
        filelist = [x for x in filelist if os.path.isfile(os.path.join(restore_dir, x)) and x.endswith('.pth.tar')]
        if len(filelist) > 0:
            filelist.sort(key=lambda fn: os.path.getmtime(os.path.join(restore_dir, fn)), reverse=True)
            snapshot = os.path.join(restore_dir, filelist[0])
        else:
            snapshot = ''

    if os.path.isfile(snapshot):
        print("=> loading checkpoint '{}'".format(snapshot))
        checkpoint = torch.load(snapshot)
        try:
            if istrain:
                args.current_epoch = checkpoint['epoch'] + 1
                args.global_counter = checkpoint['global_counter'] + 1
                if including_opt:
                    optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(snapshot, checkpoint['epoch']))
        except KeyError:
            print("KeyError")
            _model_load(model, checkpoint)
        print("=> loaded checkpoint '{}'".format(snapshot))
    else:
        print("=> no checkpoint found at '{}'".format(snapshot))
        sys.exit(0)


def _model_load(model, pretrained_dict):
    model_dict = model.state_dict()
    if model_dict.keys()[0].startswith('module.'):
        pretrained_dict = {'module.' + k: v for k, v in pretrained_dict.items()}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    print("Weights cannot be loaded:")
    print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


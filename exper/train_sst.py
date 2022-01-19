import sys

sys.path.append('../')
import os
import time
import json
import datetime
import torch
from tensorboardX import SummaryWriter

from engine.engine_train import get_model_optimizer, opts, save_checkpoint, reproducibility_set, log_init, \
    warmup_init, decay_info_builder, learning_rate_decay, learning_rate_decay_dynamic, train_print, train_recording
from engine.engine_loader import data_loader
from utils import evaluate
from utils.snapshot import save_snapshot
from utils.restore import restore


def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str

    # set reproducibility parameters
    reproducibility_set(args)
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    # create save_dirs and training record file
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
        config = json.dumps(vars(args), indent=4, separators=(',', ':'))
        fw.write(config)
        fw.close()

    # construct log writer
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    # init log parameters
    log_meters = log_init(args)
    batch_time, losses, losses_cls, top1, top5, losses_so, losses_ra, losses_spa, log_head = log_meters
    with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
        fw.write(log_head)
        fw.close()

    args.device = torch.device('cuda') if args.gpus[0] >= 0 else torch.device('cpu')

    # get model and optimizer
    model, optimizer, optim_params_name, opt_modules_key = get_model_optimizer(args)

    # resume model from checkpoint
    if args.resume == 'True':
        restore(args, model, optimizer, including_opt=True)

    decay_params_idx = decay_info_builder(args, opt_modules_key)

    # warmup initial
    cos_scheduler, gra_scheduler = None, None
    if args.warmup == 'True':
        if args.warmup_fun == 'cos':
            cos_scheduler = warmup_init(args, optimizer, optim_params_name)
        if args.warmup_fun == 'gra':
            gra_scheduler = warmup_init(args, optimizer, optim_params_name)

    # recording the require grad
    require_grad_log = ''
    for name, param in model.module.named_parameters():
        if param.requires_grad:
            print("requires_grad: True ", name)
        else:
            require_grad_log += "requires_grad: False " + name + '\n'
            print("requires_grad: False ", name)
    with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
        fw.write(require_grad_log)
        fw.close()

    for p_name, g in zip(optim_params_name, optimizer.param_groups):
        print('Initial-lr:', p_name, ':', g['lr'])

    model.train()
    train_loader = data_loader(args)
    steps_per_epoch = len(train_loader)

    total_epoch = args.epoch
    current_epoch = args.current_epoch
    global_counter = args.global_counter

    end = time.time()
    max_iter = total_epoch * len(train_loader)
    print('Max iter:', max_iter)

    if args.decay_node == 'dynamic':
        decay_flag = False
        decay_once = False

    while current_epoch < total_epoch:
        losses.reset(), losses_cls.reset(), top1.reset(), top5.reset(), batch_time.reset()
        losses_ra.reset() if args.ram else None
        losses_so.reset() if 'sos' in args.mode else None
        losses_spa.reset() if args.spa_loss == 'True' else None

        # Learning rate decay
        if args.decay_node != 'dynamic':
            learning_rate_decay(decay_params_idx, optimizer, current_epoch)
        if args.decay_node == 'dynamic' and decay_flag is True:
            learning_rate_decay_dynamic(decay_params_idx, optimizer)
            total_epoch = current_epoch + 20
            decay_flag = False
            decay_once = True

        # Gradually warmup
        if args.warmup == 'True' and args.warmup_fun == 'gra' and args.dataset == 'ilsvrc':
            gra_scheduler.step(current_epoch)

        with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
            for p_name, g in zip(optim_params_name, optimizer.param_groups):
                print('Epoch:', current_epoch, p_name, ':', g['lr'])
                out_str = 'Epoch:%d, %s, %f\n' % (current_epoch, p_name, g['lr'])
                fw.write(out_str)
            fw.close()

        if args.watch_cam:
            save_flag = True

        for idx, dat in enumerate(train_loader):
            global_counter += 1
            img_path, input_img, label = dat
            input_img, label = input_img.to(args.device), label.to(args.device)
            b_s = input_img.size()[0]
            sc_maps_fo, sc_maps_so, pred_sos, gt_scm = [None] * 4

            # forward pass and cal loss
            if args.mode == 'spa':
                logits, _, _ = model(x=input_img, cur_epoch=current_epoch)
            elif args.mode == 'spa+sa':
                logits, _, _ = model(x=input_img, cur_epoch=current_epoch)
            elif args.mode == 'sos':
                logits, pred_sos, sc_maps_fo, sc_maps_so = model(x=input_img, cur_epoch=current_epoch)
            elif args.mode == 'sos+sa_v3':
                logits, pred_sos, sc_maps_fo, sc_maps_so = model(x=input_img, cur_epoch=current_epoch)
            else:
                raise Exception("[Error] Wrong training mode, please check.")
            pass
            if 'sos' in args.mode and current_epoch >= args.sos_start:
                gt_scm = model.module.get_scm(logits, label, sc_maps_fo, sc_maps_so)
                gt_scm = gt_scm.to(args.device)
            pass
            loss_params = {'cls_logits': logits, 'cls_label': label, 'pred_sos': pred_sos,
                           'gt_sos': gt_scm, 'current_epoch': current_epoch}
            loss_val, loss_cls, loss_ra, loss_so = model.module.get_loss(loss_params)
            if args.spa_loss == 'True':
                if args.spa_loss_start <= current_epoch:
                    bs, _, _, _ = logits.size()
                    gt_map = logits[torch.arange(bs), label.long(), ...]
                    loss_spa = model.module.get_sparse_loss(gt_map)
                    loss_val += loss_spa
                else:
                    loss_spa = torch.zeros_like(loss_val)
            pass

            # write into tensorboard
            writer.add_scalar('loss_val', loss_val, global_counter)
            writer.add_scalar('loss_cls', loss_cls, global_counter)

            # network parameter update
            optimizer.zero_grad()
            loss_val.backward()
            # cosine warmup
            if args.warmup == 'True' and args.warmup_fun == 'cos':
                cos_scheduler.step(current_epoch + args.batch_size / steps_per_epoch)
            pass
            optimizer.step()
            cls_logits = torch.mean(torch.mean(logits, dim=2), dim=2)

            if not args.onehot == 'True':
                prec1, prec5 = evaluate.accuracy(cls_logits.data, label.long(), topk=(1, 5))
                top1.update(prec1[0], b_s)
                top5.update(prec5[0], b_s)
            pass
            losses.update(loss_val.data, b_s)
            losses_cls.update(loss_cls.data, b_s)
            if 'sos' in args.mode:
                losses_so.update(loss_so.data, b_s)
            pass
            if args.ram:
                losses_ra.update(loss_ra.data, b_s)
            pass
            if args.spa_loss == 'True':
                losses_spa.update(loss_spa.data, b_s)
            pass
            batch_time.update(time.time() - end)
            end = time.time()

            if global_counter % args.disp_interval == 0:
                train_print(args, idx, losses, losses_cls, losses_so, losses_ra, losses_spa, top1, top5,
                total_epoch, current_epoch, steps_per_epoch, batch_time, global_counter)
                writer.add_scalar('top1', top1.avg, global_counter)
                writer.add_scalar('top5', top5.avg, global_counter)
            pass

            if args.watch_cam and save_flag is True:
                save_snapshot(args, img_path, input_img, logits, cls_logits, label, gt_scm, pred_sos, current_epoch)
                save_flag = False
            pass
        pass

        current_epoch += 1

        checkpoint_save_condition = (current_epoch % 10 == 0 and args.dataset == 'cub') or \
                                    (args.dataset == 'ilsvrc') or current_epoch == total_epoch
        if checkpoint_save_condition:
            checkpoint_save_fields = {'epoch': current_epoch, 'arch': args.arch, 'global_counter': global_counter,
                                      'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            checkpoint_save_name = '%s_epoch_%d.pth.tar' % (args.dataset, current_epoch)
            save_checkpoint(args, checkpoint_save_fields, checkpoint_save_name)
        pass

        # training recording
        if args.decay_node == 'dynamic':
            decay_flag = train_recording(args, writer, losses, losses_cls, losses_ra, losses_so, losses_spa, top1, top5,
                        current_epoch, decay_flag, decay_once)
        else:
            train_recording(args, writer, losses, losses_cls, losses_ra, losses_so, losses_spa, top1, top5, current_epoch)


if __name__ == '__main__':
    args = opts().parse()
    train(args)

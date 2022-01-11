import sys

sys.path.append('../')
import os
import time
import json
import datetime
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from engine.engine_train import get_model, opts, save_checkpoint, reproducibility_set, log_init, \
    warmup_init, set_decay_modules, lr_decay
from engine.engine_loader import data_loader
from utils import evaluate
from utils.snapshot import save_cam, save_sos, save_scm
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

    # load log parameters
    log_meters = log_init(args)
    batch_time, losses, losses_cls, top1, top5, losses_so, losses_ra, losses_spa, log_head = log_meters
    with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
        fw.write(log_head)
        fw.close()

    args.device = torch.device('cuda') if args.gpus[0] >= 0 else torch.device('cpu')

    # get model and optimizer
    model, optimizer, params_id_list = get_model(args)

    # resume model from checkpoint
    if args.resume == 'True':
        if args.freeze == 'True':
            restore(args, model, optimizer, including_opt=False)
        else:
            restore(args, model, optimizer, including_opt=True)

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

    for p_name, g in zip(params_id_list, optimizer.param_groups):
        print('Initial-lr:', p_name, ':', g['lr'])

    # warmup initial
    cos_scheduler, gra_scheduler = None, None
    if args.warmup == 'True':
        if args.warmup_fun == 'cos':
            cos_scheduler = warmup_init(args, optimizer, params_id_list)
        if args.warmup_fun == 'gra':
            gra_scheduler = warmup_init(args, optimizer, params_id_list)

    # Specify the decay module
    decay_params = set_decay_modules(args.decay_module)

    # set training mode
    model.train()
    # load train loader
    train_loader = data_loader(args)

    total_epoch = args.epoch
    current_epoch = args.current_epoch
    global_counter = args.global_counter

    end = time.time()
    max_iter = total_epoch * len(train_loader)
    print('Max iter:', max_iter)

    decay_flag = False
    decay_once = False
    decay_count = 0

    while current_epoch < total_epoch:
        steps_per_epoch = len(train_loader)
        model.train()

        losses.reset()
        losses_cls.reset()
        top1.reset()
        top5.reset()
        batch_time.reset()
        if 'sos' in args.mode:
            losses_so.reset()
        if args.ram:
            losses_ra.reset()
        if args.spa_loss == 'True':
            losses_spa.reset()

        #  learning rate decay.
        return_list = lr_decay(args, decay_params, decay_count, decay_flag, decay_once,
                               optimizer, current_epoch, gra_scheduler)
        if len(return_list) > 4:
            total_epoch, optimizer, decay_count, decay_once, gra_scheduler = return_list
        else:
            optimizer, decay_count, decay_once, gra_scheduler = return_list

        with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
            for p_name, g in zip(params_id_list, optimizer.param_groups):
                print('Epoch:', current_epoch, p_name, ':', g['lr'])
                out_str = 'Epoch:%d, %s, %f\n' % (current_epoch, p_name, g['lr'])
                fw.write(out_str)

        save_flag = True  # 'save_cam' during training.
        watch_trans_img, watch_cam, watch_cls_logits, watch_img_path, \
        watch_label, watch_sos, watch_gt = [None] * 7

        for idx, dat in enumerate(train_loader):
            # len(train_loader) = how many batchs in train_loader
            global_counter += 1

            img_path, img, label = dat
            input_img = img  # (bs, 3, h, w)
            input_img, label = input_img.to(args.device), label.to(args.device)

            sc_maps_fo = None
            sc_maps_so = None
            pred_sos = None
            gt_scm = None

            # forward pass
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

            if 'sos' in args.mode and current_epoch >= args.sos_start:
                gt_scm = model.module.get_scm(logits, label, sc_maps_fo, sc_maps_so)
                gt_scm = gt_scm.to(args.device)

            loss_params = {'cls_logits': logits, 'cls_label': label, 'pred_sos': pred_sos,
                           'gt_sos': gt_scm, 'current_epoch': current_epoch}

            loss_val, loss_cls, loss_ra, loss_so = model.module.get_loss(loss_params)

            if args.spa_loss == 'True' and args.spa_loss_start <= current_epoch:
                bs, _, _, _ = logits.size()
                gt_map = logits[torch.arange(bs), label.long(), ...]
                loss_spa = model.module.get_sparse_loss(gt_map)
                loss_val += loss_spa

            # write into tensorboard
            writer.add_scalar('loss_val', loss_val, global_counter)
            writer.add_scalar('loss_cls', loss_cls, global_counter)

            # network parameter update
            optimizer.zero_grad()
            loss_val.backward()

            # for cosine warmup
            if args.warmup == 'True' and args.warmup_fun == 'cos' and decay_flag is False:
                cos_scheduler.step(current_epoch + args.batch_size / steps_per_epoch)

            optimizer.step()

            # GAP
            cls_logits = torch.mean(torch.mean(logits, dim=2), dim=2)

            if not args.onehot == 'True':
                prec1, prec5 = evaluate.accuracy(cls_logits.data, label.long(), topk=(1, 5))
                top1.update(prec1[0], input_img.size()[0])
                top5.update(prec5[0], input_img.size()[0])

            losses.update(loss_val.data, input_img.size()[0])
            losses_cls.update(loss_cls.data, input_img.size()[0])

            if 'sos' in args.mode:
                losses_so.update(loss_so.data, input_img.size()[0])

            if args.ram:
                losses_ra.update(loss_ra.data, input_img.size()[0])

            if args.spa_loss == 'True' and current_epoch >= args.spa_loss_start:
                losses_spa.update(loss_spa.data, input_img.size()[0])

            batch_time.update(time.time() - end)

            end = time.time()

            if global_counter % args.disp_interval == 0:
                # Calculate ETA
                eta_seconds = ((total_epoch - current_epoch) * steps_per_epoch +
                               (steps_per_epoch - idx)) * batch_time.avg
                eta_str = "{:0>8}".format(str(datetime.timedelta(seconds=int(eta_seconds))))
                eta_seconds_epoch = steps_per_epoch * batch_time.avg
                eta_str_epoch = "{:0>8}".format(str(datetime.timedelta(seconds=int(eta_seconds_epoch))))
                log_output = 'Epoch: [{0}][{1}/{2}] \t ' \
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t ' \
                             'ETA {eta_str}({eta_str_epoch})\t ' \
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t ' \
                             'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t ' \
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t ' \
                             'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(current_epoch,
                                                                               global_counter % len(train_loader),
                                                                               len(train_loader), batch_time=batch_time,
                                                                               eta_str=eta_str,
                                                                               eta_str_epoch=eta_str_epoch, loss=losses,
                                                                               loss_cls=losses_cls,
                                                                               top1=top1, top5=top5)

                if 'sos' in args.mode:
                    log_output += 'Loss_so {loss_so.val:.4f} ({loss_so.avg:.4f})\t'.format(loss_so=losses_so)

                if args.ram:
                    log_output += 'Loss_ra {loss_ra.val:.4f} ({loss_ra.avg:.4f})\t'.format(loss_ra=losses_ra)

                if args.spa_loss == 'True':
                    log_output += 'Loss_spa {loss_spa.val:.4f} ({loss_spa.avg:.4f})\t'.format(loss_spa=losses_spa)

                print(log_output)
                writer.add_scalar('top1', top1.avg, global_counter)
                writer.add_scalar('top5', top5.avg, global_counter)

            if args.watch_cam:
                if args.dataset == 'cub':
                    want_im = 'Scarlet_Tanager_0083_138500'
                elif args.dataset == 'ilsvrc':
                    want_im = 'n02488291_5176'
                else:
                    raise Exception('[Error] Invalid dataset!')
                for idx, im in enumerate(img_path):
                    if want_im in im and save_flag is True:
                        watch_trans_img = input_img[idx]
                        watch_cam = F.relu(logits)[idx]
                        watch_cls_logits = cls_logits[idx]
                        watch_img_path = im
                        watch_label = label.long()[idx]
                        if 'sos' in args.mode and current_epoch >= args.sos_start:
                            watch_gt = [gt_scm[idx]]
                            # watch_scm = [(sc_maps_fo[-2][idx], sc_maps_fo[-1][idx]),
                            #              (sc_maps_so[-2][idx], sc_maps_so[-1][idx])]
                            watch_sos = pred_sos[idx]
                            watch_sos = [torch.sigmoid(watch_sos)] if args.sos_loss_method == 'BCE' else [watch_sos]
                        save_flag = False

        # vis cam during training
        if args.watch_cam:
            save_cam(args, watch_trans_img, watch_cam, watch_cls_logits, watch_img_path, watch_label, current_epoch)
            if 'sos' in args.mode and current_epoch >= args.sos_start:
                save_sos(args, watch_trans_img, watch_sos, watch_img_path, current_epoch, suffix='sos')
                save_sos(args, watch_trans_img, watch_gt, watch_img_path, current_epoch, suffix='gt_sos')
                # save_scm(args, watch_trans_img, watch_scm, watch_img_path, current_epoch, suffix='sc_4+5')

        current_epoch += 1
        if (current_epoch % 10 == 0 and args.dataset == 'cub') or (
                args.dataset == 'ilsvrc') or current_epoch == total_epoch:
            save_checkpoint(args,
                            {'epoch': current_epoch,
                             'arch': args.arch,
                             'global_counter': global_counter,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }, is_best=False,
                            filename='%s_epoch_%d.pth.tar'
                                     % (args.dataset, current_epoch))

        # save training record
        with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
            log_output = '{} \t {:.4f} \t {:.3f} \t {:.3f} \t {:.3f} \t'.format(current_epoch, losses.avg,
                                                                                losses_cls.avg,
                                                                                top1.avg, top5.avg)
            writer.add_scalar('loss_epoch', losses.avg, current_epoch)
            writer.add_scalar('cls_loss_epoch', losses_cls.avg, current_epoch)

            if args.ram:
                log_output += '{:.4f} \t'.format(losses_ra.avg)
                writer.add_scalar('ram_loss', losses_ra.avg, current_epoch)
                if args.dataset == 'cub':
                    if args.decay_points == 'none' and decay_flag is False and losses_ra.avg >= 0.30 and current_epoch >= 80:
                        decay_flag = True
                    elif args.decay_points == 'none' and decay_flag is False and current_epoch >= 99:
                        decay_flag = True
                if args.dataset == 'ilsvrc':
                    pass

            if 'sos' in args.mode:
                log_output += '{:.4f} \t'.format(losses_so.avg)
                writer.add_scalar('sos_loss', losses_so.avg, current_epoch)

            if args.spa_loss == 'True':
                log_output += '{:.4f} \t'.format(losses_spa.avg)
                writer.add_scalar('spa_loss', losses_spa.avg, current_epoch)

            log_output += '\n'
            fw.write(log_output)
            fw.close()

        losses.reset()
        losses_cls.reset()

        if 'sos' in args.mode:
            losses_so.reset()

        if args.ram:
            losses_ra.reset()

        if args.spa_loss == 'True':
            losses_spa.reset()

        top1.reset()
        top5.reset()


if __name__ == '__main__':
    args = opts().parse()
    train(args)

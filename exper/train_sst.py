import sys

sys.path.append('../')
import os
import time
import json
import datetime
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from utils import my_optim
from utils.init_train import get_model, opts, save_checkpoint, reproducibility_set, log_init, warmup_init
from utils import evaluate
from utils.loader import data_loader
from utils.save_cam_test import save_cam, save_sos, save_scm
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

    # load log parameters
    log_meters = log_init(args)
    # batch_time, losses, top1, top5, losses_so, losses_hg, losses_ra, losses_spa, log_head = log_meters
    batch_time, losses, top1, top5, losses_so, losses_ra, losses_spa, log_head = log_meters
    with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
        fw.write(log_head)

    args.device = torch.device('cuda') if args.gpus[0] >= 0 else torch.device('cpu')

    # get model parameters
    model, optimizer, op_params_list = get_model(args)

    # resume model from checkpoint
    if args.resume == 'True':
        restore(args, model, optimizer, including_opt=True)

    for p_name, g in zip(op_params_list, optimizer.param_groups):
        print('Initial-lr:', p_name, ':', g['lr'])

    # init warmup params
    if args.warmup == 'True':
        if args.warmup_fun == 'cos':
            cos_scheduler = warmup_init(args, optimizer, op_params_list)
        if args.warmup_fun == 'gra':
            gra_scheduler = warmup_init(args, optimizer, op_params_list)

    # set parameter_lr for decay and increasing
    # 0,1 / 2,3 / 4,5 / 6,7 => bb / cls / sos / sa, 'all' for all
    # only used for ILSVRC now.
    if args.dataset == 'ilsvrc':
        decay_params = ['0,1,2,3,6,7', '0,1,2,3,6,7', None]
    elif args.dataset == 'cub':
        decay_params = ['0,1,2,3,6,7', None]
    else:
        raise Exception("Wrong dataset, please check !")

    if args.increase_lr == 'True':
        increase_params = [None]

    model.train()
    train_loader = data_loader(args)

    # construct writer
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    total_epoch = args.epoch
    global_counter = args.global_counter

    current_epoch = args.current_epoch
    # current_epoch = 5

    end = time.time()
    max_iter = total_epoch * len(train_loader)
    print('Max iter:', max_iter)
    decay_flag = False
    decay_once = False
    increase_once = False
    decay_count = 0
    increase_count = 0

    while current_epoch < total_epoch:
        model.train()
        losses.reset()
        if 'sos' in args.mode:
            losses_so.reset()
        # if 'hinge' in args.mode:
        #     losses_hg.reset()
        # if 'rcst' in args.mode or 'sst' in args.mode:
        #     losses_rcst.reset()
        if args.ram:
            losses_ra.reset()
        if args.spa_loss == 'True':
            losses_spa.reset()
        top1.reset()
        top5.reset()
        batch_time.reset()

        if args.warmup == 'True':  # with warmup
            if args.dataset == 'cub':
                if args.decay_points == 'none':
                    if decay_flag is True and decay_once is False:
                        if my_optim.reduce_lr(args, optimizer, current_epoch):
                            total_epoch = current_epoch + 20
                            decay_once = True
                            if args.warmup_fun == 'gra':
                                gra_scheduler.update_optimizer(optimizer, current_epoch)
                else:
                    raise Exception("On CUB200, u should specify decay_points = none !")
                if args.warmup_fun == 'gra':
                    gra_scheduler.step(current_epoch)
            if args.dataset == 'ilsvrc':
                if my_optim.reduce_lr(args, optimizer, current_epoch):
                    if args.warmup_fun == 'gra':
                        gra_scheduler.update_optimizer(optimizer, current_epoch)
                if args.warmup_fun == 'gra':
                    gra_scheduler.step(current_epoch)
        else:  # without warmup
            if args.dataset == 'cub':
                if args.decay_points == 'none':
                    if decay_flag is True and decay_once is False:
                        decay_str = decay_params[decay_count]
                        if my_optim.reduce_lr(args, optimizer, current_epoch, decay_params=decay_str):
                            total_epoch = current_epoch + 20
                            decay_once = True
                else:
                    if my_optim.reduce_lr(args, optimizer, current_epoch):
                        decay_once = True
            if args.dataset == 'ilsvrc':
                decay_str = decay_params[decay_count]
                if my_optim.reduce_lr(args, optimizer, current_epoch, decay_params=decay_str):
                    decay_once = True
                    decay_count += 1

        # Increasing learning rate only once.
        if args.increase_lr == 'True' and args.dataset == 'ilsvrc':
            if decay_once is True and increase_once is False:
                increase_str = increase_params[increase_count]
                if my_optim.increase_lr(args, optimizer, current_epoch, increase_params=increase_str):
                    decay_once = False
                    increase_once = True
                    increase_count += 1

        with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
            for p_name, g in zip(op_params_list, optimizer.param_groups):
                print('Epoch:', current_epoch, p_name, ':', g['lr'])
                out_str = 'Epoch:%d, %s, %f\n' % (current_epoch, p_name, g['lr'])
                fw.write(out_str)

        steps_per_epoch = len(train_loader)

        save_flag = True

        for idx, dat in enumerate(train_loader):  # len(train_loader) = how many batchs in train_loader
            global_counter += 1

            img_path, img, label = dat
            input_img = img  # (bs, 3, h, w)
            input_img, label = input_img.to(args.device), label.to(args.device)

            logits = None
            # logits_hg = None
            sc_maps_fo = None
            sc_maps_so = None
            pred_sos = None
            gt_scm = None

            # forward pass
            if args.mode == 'spa':
                logits, _, _ = model(x=input_img, cur_epoch=current_epoch)
            # if args.mode == 'spa+hinge':
            #     logits, logits_hg, _, _ = model(x=input_img, cur_epoch=current_epoch)
            if args.mode == 'spa+sa':
                logits, _, _ = model(x=input_img, cur_epoch=current_epoch)
            if args.mode == 'sos':
                logits, pred_sos, sc_maps_fo, sc_maps_so = model(x=input_img, cur_epoch=current_epoch)
            if 'sos+sa' in args.mode:
                logits, pred_sos, sc_maps_fo, sc_maps_so = model(x=input_img, cur_epoch=current_epoch)
            # if args.mode == 'mc_sos':
            #     logits, pred_sos, sc_maps_fo, sc_maps_so = model(x=input_img, cur_epoch=current_epoch)
            # if args.mode == 'rcst':
            #     logits, sc_maps_fo, sc_maps_so, rcst_map = model(input_img, cur_epoch=current_epoch)
            # if args.mode == 'sst':
            #     logits, pred_sos, sc_maps_fo, sc_maps_so, rcst_map = model(input_img, cur_epoch=current_epoch)
            # if args.mode == 'rcst+sa':
            #     logits, sc_maps_fo, sc_maps_so, rcst_map = model(input_img, cur_epoch=current_epoch)
            # if args.mode == 'sst+sa':
            #     pass

            if 'sos' in args.mode and current_epoch >= args.sos_start:
                gt_scm = model.module.get_scm(logits, label, sc_maps_fo, sc_maps_so)
                gt_scm = gt_scm.to(args.device)

            # if 'rcst' in args.mode and args.rcst_signal == 'scm' and current_epoch >= args.rcst_start:
            #     gt_scm = model.module.get_scm(logits, label, sc_maps_fo, sc_maps_so)
            #     gt_scm = gt_scm.to(args.device)
            # Using SCM or SOS_MAP or ORI_IMG to supervise the RCST branch
            # if ('rcst' in args.mode or 'sst' in args.mode) and current_epoch >= args.rcst_start:
            #     if args.rcst_signal == 'scm':
            #         gt_rcst = model.module.get_masked_obj(gt_scm, input_img)
            #     elif ('sst' in args.mode) and args.rcst_signal == 'sos':
            #         gt_rcst = model.module.get_masked_obj(pred_sos, input_img)
            #     else:
            #         gt_rcst = input_img
            #     gt_rcst = gt_rcst.to(args.device)

            # get loss
            # loss_val, loss_ra, loss_so, loss_rc = model.module.get_loss(logits, label, pred_sos, gt_scm, rcst_map,
            #                                                             gt_rcst, epoch=current_epoch)

            # loss_params = {'cls_logits': logits, 'hg_logits': logits_hg, 'cls_label': label,
            #                'pred_sos': pred_sos, 'gt_sos': gt_scm, 'current_epoch': current_epoch}
            # loss_val, loss_ra, loss_so, loss_hg, loss_spa = model.module.get_loss(loss_params)
            loss_params = {'cls_logits': logits, 'cls_label': label,
                           'pred_sos': pred_sos, 'gt_sos': gt_scm, 'current_epoch': current_epoch}
            loss_val, loss_ra, loss_so = model.module.get_loss(loss_params)

            if args.spa_loss == 'True' and args.spa_loss_start <= current_epoch:
                bs, _, _, _ = logits.size()
                gt_map = logits[torch.arange(bs), label.long(), ...]
                loss_spa = model.module.get_sparse_loss(gt_map)
                loss_val += loss_spa

            # write into tensorboard
            writer.add_scalar('loss_val', loss_val, global_counter)

            # network parameter update
            optimizer.zero_grad()
            loss_val.backward()

            if args.warmup == 'True' and args.warmup_fun == 'cos' and decay_flag is False:
                cos_scheduler.step(current_epoch + args.batch_size / steps_per_epoch)

            optimizer.step()

            # GAP or TAP
            # cls_logits = model.module.thr_avg_pool(
            #     logits) if (args.use_tap == 'True') and (args.tap_start <= current_epoch) else torch.mean(
            #     torch.mean(logits, dim=2), dim=2)

            # GAP
            cls_logits =  torch.mean(torch.mean(logits, dim=2), dim=2)

            # hg_cls_logits = torch.mean(torch.mean(logits_hg, dim=2), dim=2) if 'hinge' in args.mode else None

            if not args.onehot == 'True':
                prec1, prec5 = evaluate.accuracy(cls_logits.data, label.long(), topk=(1, 5))
                top1.update(prec1[0], input_img.size()[0])
                top5.update(prec5[0], input_img.size()[0])

            losses.update(loss_val.data, input_img.size()[0])
            if 'sos' in args.mode:
                losses_so.update(loss_so.data, input_img.size()[0])

            # if 'hinge' in args.mode:
            #     losses_hg.update(loss_hg.data, input_img.size()[0])
            # if 'rcst' in args.mode or 'sst' in args.mode:
            #     losses_rcst.update(loss_rc.data, input_img.size()[0])

            if args.ram:
                losses_ra.update(loss_ra.data, input_img.size()[0])

            if args.spa_loss == 'True' and current_epoch>=args.spa_loss_start:
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
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t ' \
                             'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(current_epoch,
                                                                               global_counter % len(train_loader),
                                                                               len(train_loader), batch_time=batch_time,
                                                                               eta_str=eta_str,
                                                                               eta_str_epoch=eta_str_epoch, loss=losses,
                                                                               top1=top1, top5=top5)
                if 'sos' in args.mode:
                    log_output += 'Loss_so {loss_so.val:.4f} ({loss_so.avg:.4f})\t'.format(loss_so=losses_so)

                # if 'hinge' in args.mode:
                #     log_output += 'Loss_hg {loss_hg.val:.4f} ({loss_hg.avg:.4f})\t'.format(loss_hg=losses_hg)
                # if 'rcst' in args.mode or 'sst' in args.mode:
                #     log_output += 'Loss_rc {loss_rc.val:.4f} ({loss_rc.avg:.4f})\t'.format(loss_rc=losses_rcst)

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
                    want_im = 'n01440764_2574'
                else:
                    raise Exception('[Error] Invalid dataset!')
                for idx, im in enumerate(img_path):
                    if want_im in im and save_flag is True:
                        watch_trans_img = input_img[idx]
                        watch_cam = F.relu(logits)[idx]
                        watch_cls_logits = cls_logits[idx]
                        watch_img_path = im
                        watch_label = label.long()[idx]
                        # if 'hinge' in args.mode:
                        #     watch_hg_cam = F.relu(logits_hg)[idx]
                        #     watch_hg_cls_logits = hg_cls_logits[idx]
                        if 'sos' in args.mode and current_epoch >= args.sos_start:
                            watch_gt = [gt_scm[idx]]
                            watch_scm = [(sc_maps_fo[-2][idx], sc_maps_fo[-1][idx]),
                                         (sc_maps_so[-2][idx], sc_maps_so[-1][idx])]
                            watch_sos = pred_sos[idx]
                            if 'mc_sos' in args.mode:
                                watch_sos = watch_sos[watch_label]
                            watch_sos = [torch.sigmoid(watch_sos)] if args.sos_loss_method == 'BCE' else [watch_sos]
                        save_flag = False

        # vis cam during training
        if args.watch_cam:
            save_cam(args, watch_trans_img, watch_cam, watch_cls_logits, watch_img_path, watch_label, current_epoch)
            if 'sos' in args.mode and current_epoch >= args.sos_start:
                save_sos(args, watch_trans_img, watch_sos, watch_img_path, current_epoch, suffix='sos')
                save_sos(args, watch_trans_img, watch_gt, watch_img_path, current_epoch, suffix='gt_sos')
                save_scm(args, watch_trans_img, watch_scm, watch_img_path, current_epoch, suffix='sc_4+5')
            # if 'hinge' in args.mode:
            #     save_cam(args, watch_trans_img, watch_hg_cam, watch_hg_cls_logits, watch_img_path, watch_label,
            #              current_epoch, suffix='cam_hg')

        # training end
        current_epoch += 1
        if (current_epoch % 10 == 0 and args.dataset=='cub') or (args.dataset=='ilsvrc') or current_epoch == total_epoch:
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
            log_output = '{} \t {:.4f} \t {:.3f} \t {:.3f} \t'.format(current_epoch, losses.avg, top1.avg, top5.avg)
            writer.add_scalar('loss_epoch', losses.avg, current_epoch)
            if 'sos' in args.mode:
                log_output += '{:.4f} \t'.format(losses_so.avg)
                writer.add_scalar('sos_loss', losses_so.avg, current_epoch)
            # if 'hinge' in args.mode:
            #     log_output += '{:.4f} \t'.format(losses_hg.avg)
            #     writer.add_scalar('hinge_loss', losses_hg.avg, current_epoch)
            # if 'rcst' in args.mode or 'sst' in args.mode:
            #     log_output += '{:.4f} \t'.format(losses_rcst.avg)
            if args.ram:
                log_output += '{:.4f}'.format(losses_ra.avg)
                writer.add_scalar('ram_loss', losses_ra.avg, current_epoch)

                if args.dataset == 'cub':  # while ram_loss > 0.30, learning rate decay
                    if args.decay_points == 'none' and decay_flag is False and losses_ra.avg >= 0.30 and current_epoch >= 80:
                        decay_flag = True
                    elif args.decay_points == 'none' and decay_flag is False and current_epoch >= 100:
                        decay_flag = True
                if args.dataset == 'ilsvrc':
                    pass

            if args.spa_loss == 'True':
                log_output += '{:.4f} \t'.format(losses_spa.avg)
                writer.add_scalar('spa_loss', losses_spa.avg, current_epoch)

            log_output += '\n'
            fw.write(log_output)

        losses.reset()

        if 'sos' in args.mode:
            losses_so.reset()

        # if 'hinge' in args.mode:
        #     losses_hg.reset()
        # if 'rcst' in args.mode or 'sst' in args.mode:
        #     losses_rcst.reset()

        if args.ram:
            losses_ra.reset()

        if args.spa_loss == 'True':
            losses_spa.reset()

        top1.reset()
        top5.reset()


if __name__ == '__main__':
    args = opts().parse()
    train(args)

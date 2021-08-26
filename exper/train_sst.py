import sys

sys.path.append('../')
import os
import time
import json
import datetime
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np

from utils import my_optim
from utils.init_train import get_model, opts, save_checkpoint, reproducibility_set, log_init
from utils import evaluate
from utils.loader import data_loader
from utils.save_cam_test import save_cam, save_sos, save_scm


def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str

    reproducibility_set(args)
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
        config = json.dumps(vars(args), indent=4, separators=(',', ':'))
        fw.write(config)

    log_meters = log_init(args)
    batch_time, losses, top1, top5, losses_so, losses_hg, losses_ra, log_head = log_meters

    with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
        fw.write(log_head)

    args.device = torch.device('cuda') if args.gpus[0] >= 0 else torch.device('cpu')

    # get model parameters
    model, optimizer = get_model(args)
    model.train()
    train_loader = data_loader(args)

    # construct writer
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    total_epoch = args.epoch
    global_counter = args.global_counter  # 全局计数器
    current_epoch = args.current_epoch  # 当前轮次
    end = time.time()
    max_iter = total_epoch * len(train_loader)
    print('Max iter:', max_iter)

    while current_epoch < total_epoch:
        model.train()

        losses.reset()
        if 'sos' in args.mode:
            losses_so.reset()
        if 'hinge' in args.mode:
            losses_hg.reset()
        # if 'rcst' in args.mode or 'sst' in args.mode:
        #     losses_rcst.reset()
        if args.ram:
            losses_ra.reset()
        top1.reset()
        top5.reset()
        batch_time.reset()

        res = my_optim.reduce_lr(args, optimizer, current_epoch)
        if res:
            with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
                for g in optimizer.param_groups:
                    out_str = 'Epoch:%d, %f\n' % (current_epoch, g['lr'])
                    fw.write(out_str)

        steps_per_epoch = len(train_loader)

        save_flag = True

        for idx, dat in enumerate(train_loader):
            global_counter += 1

            img_path, img, label = dat
            input_img = img  # (bs, 3, h, w)
            input_img, label = input_img.to(args.device), label.to(args.device)

            logits = None
            logits_hg = None
            sc_maps_fo = None
            sc_maps_so = None
            pred_sos = None
            gt_scm = None

            # forward pass
            if args.mode == 'spa':
                logits, _, _ = model(x=input_img, cur_epoch=current_epoch)
            if args.mode == 'spa+hinge':
                logits, logits_hg, _, _ = model(x=input_img, cur_epoch=current_epoch)
            if args.mode == 'spa+sa':
                logits, _, _ = model(x=input_img, cur_epoch=current_epoch)
            if args.mode == 'sos':
                logits, pred_sos, sc_maps_fo, sc_maps_so = model(x=input_img, cur_epoch=current_epoch)
            if args.mode == 'sos+sa':
                logits, pred_sos, sc_maps_fo, sc_maps_so = model(x=input_img, cur_epoch=current_epoch)
            if args.mode == 'mc_sos':
                logits, pred_sos, sc_maps_fo, sc_maps_so = model(x=input_img, cur_epoch=current_epoch)

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

            loss_params = {'cls_logits': logits, 'hg_logits': logits_hg, 'cls_label': label,
                           'pred_sos': pred_sos, 'gt_sos': gt_scm, 'current_epoch': current_epoch}
            loss_val, loss_ra, loss_so, loss_hg = model.module.get_loss(loss_params)

            # write into tensorboard
            writer.add_scalar('loss_val', loss_val, global_counter)

            # network parameter update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # global average pooling
            cls_logits = model.module.thr_avg_pool(
                logits) if (args.use_tap == 'True') and (args.tap_start <= current_epoch) else torch.mean(
                torch.mean(logits, dim=2), dim=2)

            hg_cls_logits = torch.mean(torch.mean(logits_hg, dim=2), dim=2) if 'hinge' in args.mode else None

            if not args.onehot == 'True':
                prec1, prec5 = evaluate.accuracy(cls_logits.data, label.long(), topk=(1, 5))
                top1.update(prec1[0], input_img.size()[0])
                top5.update(prec5[0], input_img.size()[0])

            losses.update(loss_val.data, input_img.size()[0])
            if 'sos' in args.mode:
                losses_so.update(loss_so.data, input_img.size()[0])
            if 'hinge' in args.mode:
                losses_hg.update(loss_hg.data, input_img.size()[0])
            # if 'rcst' in args.mode or 'sst' in args.mode:
            #     losses_rcst.update(loss_rc.data, input_img.size()[0])
            if args.ram:
                losses_ra.update(loss_ra.data, input_img.size()[0])
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
                if 'hinge' in args.mode:
                    log_output += 'Loss_hg {loss_hg.val:.4f} ({loss_hg.avg:.4f})\t'.format(loss_hg=losses_hg)
                # if 'rcst' in args.mode or 'sst' in args.mode:
                #     log_output += 'Loss_rc {loss_rc.val:.4f} ({loss_rc.avg:.4f})\t'.format(loss_rc=losses_rcst)
                if args.ram:
                    log_output += 'Loss_ra {loss_ra.val:.4f} ({loss_ra.avg:.4f})\t'.format(loss_ra=losses_ra)
                print(log_output)
                writer.add_scalar('top1', top1.avg, global_counter)
                writer.add_scalar('top5', top5.avg, global_counter)

            if args.watch_cam:
                want_im = 'Scarlet_Tanager_0083_138500'
                for idx, im in enumerate(img_path):
                    if want_im in im and save_flag is True:
                        watch_trans_img = input_img[idx]
                        watch_cam = F.relu(logits)[idx]
                        watch_cls_logits = cls_logits[idx]
                        watch_img_path = im
                        watch_label = label.long()[idx]
                        if 'hinge' in args.mode:
                            watch_hg_cam = F.relu(logits_hg)[idx]
                            watch_hg_cls_logits = hg_cls_logits[idx]
                        if 'sos' in args.mode and current_epoch >= args.sos_start:
                            watch_gt = [gt_scm[idx]]
                            watch_scm = [(sc_maps_fo[-2][idx], sc_maps_fo[-1][idx]),
                                         (sc_maps_so[-2][idx], sc_maps_so[-1][idx])]
                            if args.sos_seg_method == 'BC':  # sigmoid & threshold
                                watch_sos = pred_sos[idx]
                                if 'mc_sos' in args.mode:
                                    watch_sos = watch_sos[watch_label]
                                watch_sos = [torch.sigmoid(watch_sos)]
                            else:
                                watch_sos = [pred_sos[idx]]
                        save_flag = False

        # vis cam during training
        if args.watch_cam:
            save_cam(args, watch_trans_img, watch_cam, watch_cls_logits, watch_img_path, watch_label, current_epoch)
            if 'sos' in args.mode and current_epoch >= args.sos_start:
                save_sos(args, watch_trans_img, watch_sos, watch_img_path, current_epoch, suffix='sos')
                save_sos(args, watch_trans_img, watch_gt, watch_img_path, current_epoch, suffix='gt_sos')
                save_scm(args, watch_trans_img, watch_scm, watch_img_path, current_epoch, suffix='sc_4+5')
            if 'hinge' in args.mode:
                save_cam(args, watch_trans_img, watch_hg_cam, watch_hg_cls_logits, watch_img_path, watch_label,
                         current_epoch, suffix='cam_hg')

        # training end
        current_epoch += 1
        if current_epoch % 10 == 0:
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
            if 'sos' in args.mode:
                log_output += '{:.4f} \t'.format(losses_so.avg)
            if 'hinge' in args.mode:
                log_output += '{:.4f} \t'.format(losses_hg.avg)
            # if 'rcst' in args.mode or 'sst' in args.mode:
            #     log_output += '{:.4f} \t'.format(losses_rcst.avg)
            if args.ram:
                log_output += '{:.4f}'.format(losses_ra.avg)
            log_output += '\n'
            fw.write(log_output)

        losses.reset()
        # if 'sos' in args.mode or 'sst' in args.mode:
        if 'sos' in args.mode:
            losses_so.reset()
        if 'hinge' in args.mode:
            losses_hg.reset()
        # if 'rcst' in args.mode or 'sst' in args.mode:
        #     losses_rcst.reset()
        if args.ram:
            losses_ra.reset()
        top1.reset()
        top5.reset()


if __name__ == '__main__':
    args = opts().parse()
    train(args)

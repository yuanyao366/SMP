from config import params
from torch import nn, optim
import os
import sys
from models import c3d, r3d, r21d, sscn
from datasets.predict_dataset import PredictDataset
from datasets import video_transforms
from torchvision import transforms
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import random
import numpy as np
from tensorboardX import SummaryWriter
from visualze import *
from utils import Logger
import argparse
import torch.nn.functional as F

multi_gpu = 1
start_epoch = 1
ckpt = None
params['batch_size'] = 8
params['num_workers'] = 4
params['dataset'] = '/home/Dataset/UCF-101-origin'
params['data'] = 'UCF-101'
learning_rate = 0.01


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Finsert_MSEloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, clip_label, step_label):
        loss_batch_list = []
        output_batch_list = []
        clip_label_batch_list = []
        sample_step_list = [1, 2, 4, 8]
        batch_size = step_label.size(0)
        for i in range(batch_size):
            step_label_i = step_label[i].item()
            sample_len = sample_step_list[step_label_i] * 16
            clip_label_i = clip_label[i, :, :sample_len, :, :]
            output_i = output[step_label_i][i]
            loss_i = torch.mean(torch.pow((output_i - clip_label_i), 2))
            loss_batch_list.append(loss_i)
            clip_label_batch_list.append(clip_label_i)
            output_batch_list.append(output_i)
        loss_batch = torch.stack(loss_batch_list)
        loss = torch.mean(loss_batch)
        return loss, output_batch_list, clip_label_batch_list


class Motion_MSEloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, clip_label, motion_mask):
        z = torch.pow((output - clip_label), 2)
        loss = torch.mean(motion_mask * z)
        return loss


class Motion_MSEloss_NFGT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, clip_label, motion_mask, recon_flags):
        loss_batch_list = []
        batch_size = output.size(0)
        for i in range(batch_size):
            Tinds = torch.nonzero(recon_flags[i]).squeeze()
            z = torch.pow((output[i][:, Tinds, :, :] - clip_label[i][:, Tinds, :, :]), 2)
            loss_i = torch.mean(z * motion_mask[i][:, Tinds, :, :])
            loss_batch_list.append(loss_i)
        loss_batch = torch.stack(loss_batch_list)
        loss = torch.mean(loss_batch)
        return loss


def Cam_Mask(feats, fc8, recon_clip, step_label, act_fun):
    fb, fc, ft, fh, fw = feats.size()
    fc8_outch, fc8_inch = fc8.size()
    cam = torch.matmul(fc8, feats.reshape(fb, fc, -1))
    cam = cam.reshape(fb, fc8_outch, ft, fh, fw)
    cam_cls = []
    for i in range(len(step_label)):
        cam_cls.append(cam[i, step_label[i], :, :, :])
    cam_cls = torch.stack(cam_cls, dim=0)
    b, t, h, w = cam_cls.size()
    mm_r = cam_cls.reshape(b, t, -1)
    mm_r_min = mm_r.min(dim=2, keepdim=True)[0]
    mm_r_max = mm_r.max(dim=2, keepdim=True)[0]
    mm_rs = (mm_r - mm_r_min) / (mm_r_max - mm_r_min)
    mm_rs = mm_rs.reshape(b, t, h, w)
    mm_rsl = mm_rs * 1.2 + 0.8
    rb, rc, rt, rh, rw = recon_clip.size()
    mm_rslu = F.interpolate(mm_rsl.unsqueeze(dim=1), size=(rt, rh, rw), mode='trilinear', align_corners=False)
    mm_rsluc = torch.cat([mm_rslu] * 3, dim=1)
    return mm_rsluc


def Mask_lambda(epoch, lambda_str, max_epoch=300):
    # print(lambda_str)
    if lambda_str == 'exp':
        if max_epoch <= 100:
            s = 25
        else:
            s = 75
        mask_lambda = np.exp((epoch - max_epoch) / s)
    elif lambda_str == 'log':
        s = 15
        mask_lambda = (np.log(epoch / s + 0.01) - np.log(0.01)) / (np.log(max_epoch / s + 0.01) - np.log(0.01))
    elif lambda_str == 'sigmoid':
        s = 75
        mask_lambda = 1 / (1 + np.exp((max_epoch / 2 - epoch) / s))
    elif lambda_str == 'plinear':
        s = 75
        mask_lambda = min(1.0, np.ceil(epoch / s) / max_epoch * s)
    elif lambda_str == 'linear':
        mask_lambda = epoch / max_epoch
    elif lambda_str == 'w1':
        mask_lambda = 1
    elif lambda_str == 'w0':
        mask_lambda = 0
    return float(mask_lambda)


def train(train_loader, model, criterion_MSE, criterion_CE, optimizer, epoch, writer, args=None):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_recon = AverageMeter()
    losses = AverageMeter()
    encoder_cls_head_keys = args.enc_head.split('_')
    losses_class = {}
    acc = {}
    total_cls_loss = {}
    correct_cnt ={}
    total_cls_cnt = {}
    correct_cls_cnt = {}
    for key in encoder_cls_head_keys:
        losses_class[key] = AverageMeter()
        acc[key] = AverageMeter()
        total_cls_loss[key] = 0.0
        correct_cnt[key] = 0
        total_cls_cnt[key] = torch.zeros(4)
        correct_cls_cnt[key] = torch.zeros(4)
    model.train()
    end = time.time()

    for step, (sample_clip, recon_clip, step_label, recon_rate, motion_mask, recon_flags) in enumerate(train_loader):
        data_time.update(time.time() - end)

        clip_input = sample_clip.cuda()
        clip_label = recon_clip.cuda()
        step_label = step_label.cuda()
        recon_rate = recon_rate.cuda()
        motion_mask = motion_mask.cuda()
        recon_flags = recon_flags.cuda()

        clip_output, step_output, feat_output1 = model(clip_input)
        if args.mask_name == 'cam':
            fc8 = dict(model.named_parameters())['module.fc8_c5.weight'].detach()
            feat_mask = Cam_Mask(feat_output1.detach(), fc8, recon_clip, step_label, act_fun=args.mask_act)
        elif args.mask_name == 'patch':
            feat_mask = 0

        mask_lambda = Mask_lambda(epoch, lambda_str=args.mask_w_fun, max_epoch=args.epochs)
        mask = mask_lambda * feat_mask + (1 - mask_lambda) * motion_mask
        loss_recon = criterion_MSE(clip_output, clip_label, mask, recon_flags)
        loss_class = {}
        for key in encoder_cls_head_keys:
            loss_class[key] = criterion_CE(step_output[key], step_label)
        if encoder_cls_head_keys == ['c5', 'c4', 'c3', 'c2', 'c1']:
            loss = loss_recon + loss_class['c5'] * 0.1 + 0.1 * (loss_class['c4'] + loss_class['c3'] + loss_class['c2'] + loss_class['c1']) / 4
        elif encoder_cls_head_keys == ['c5', 'c4', 'c3', 'c2']:
            loss = loss_recon + loss_class['c5'] * 0.1 + 0.1 * (loss_class['c4'] + loss_class['c3'] + loss_class['c2']) / 3
        elif encoder_cls_head_keys == ['c5', 'c4', 'c3']:
            loss = loss_recon + loss_class['c5'] * 0.1 + 0.1 * (loss_class['c4'] + loss_class['c3']) / 2
        elif encoder_cls_head_keys == ['c5', 'c4']:
            loss = loss_recon + loss_class['c5'] * 0.1 + 0.1 * loss_class['c4']
        else:
            loss = loss_recon + loss_class['c5'] * 0.1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        losses_recon.update(loss_recon.item(), clip_input.size(0))
        losses.update(loss.item(), clip_input.size(0))
        for key in encoder_cls_head_keys:
            losses_class[key].update(loss_class[key].item(), clip_input.size(0))
            prec_class = accuracy(step_output[key].data, step_label, topk=(1,))[0]
            acc[key].update(prec_class.item(), clip_input.size(0))
            total_cls_loss[key] += loss_class[key].item()
            pts = torch.argmax(step_output[key], dim=1)
            correct_cnt[key] += torch.sum(step_label == pts).item()
            for i in range(step_label.size(0)):
                total_cls_cnt[key][step_label[i]] += 1
                if step_label[i] == pts[i]:
                    correct_cls_cnt[key][pts[i]] += 1

        if (step + 1) % params['display'] == 0:
            print('-----------------------------------------------')
            p_str = "conv_lr:{} fc8_lr:{}".format(optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr'])
            print(p_str)

            p_str = "Epoch:[{0}][{1}/{2}]".format(epoch, step + 1, len(train_loader))
            print(p_str)

            p_str = "data_time:{data_time:.3f},batch time:{batch_time:.3f}".format(data_time=data_time.val,
                                                                                   batch_time=batch_time.val)
            print(p_str)

            p_str = "loss:{loss:.5f} loss_recon:{loss_recon:.5f} ".format(loss=losses.avg, loss_recon=losses_recon.avg)
            for key in encoder_cls_head_keys:
                p_str += "loss_cls_{}:{:.5f} ".format(key, losses_class[key].avg)
            print(p_str)

            p_str = ''
            for key in encoder_cls_head_keys:
                p_str += 'acc_{}:{:.3f} '.format(key, acc[key].avg)
            print(p_str)

            total_step = (epoch - 1) * len(train_loader) + step + 1
            info = {
                'loss': losses.avg,
                'loss_res': losses_recon.avg,
            }
            for key in encoder_cls_head_keys:
                info['loss_cls_{}'.format(key)] = losses_class[key].avg * 0.1
            writer.add_scalars('train/loss', info, total_step)

            for key in encoder_cls_head_keys:
                info_acc = {}
                for cls in range(correct_cls_cnt[key].size(0)):
                    acc_cls = correct_cls_cnt[key][cls] / total_cls_cnt[key][cls]
                    info_acc['cls{}'.format(cls)] = acc_cls
                info_acc['avg'] = acc[key].avg * 0.01
                writer.add_scalars('train/acc_{}'.format(key), info_acc, total_step)
            # writer.add_scalar('train/loss',losses.avg,total_step)

    for key in encoder_cls_head_keys:
        avg_cls_loss = total_cls_loss[key] / len(train_loader)
        avg_acc = correct_cnt[key] / len(train_loader.dataset)
        print('[TRAIN] loss_cls_{}: {:.3f}, acc_{}: {:.3f}'.format(key, avg_cls_loss, key, avg_acc))
        print(correct_cls_cnt[key])
        print(total_cls_cnt[key])
        print(correct_cls_cnt[key] / total_cls_cnt[key])




def validation(val_loader, model, criterion_MSE, criterion_CE, optimizer, epoch, args=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_recon = AverageMeter()
    losses = AverageMeter()
    encoder_cls_head_keys = args.enc_head.split('_')
    losses_class = {}
    acc = {}
    total_cls_loss = {}
    correct_cnt = {}
    total_cls_cnt = {}
    correct_cls_cnt = {}
    for key in encoder_cls_head_keys:
        losses_class[key] = AverageMeter()
        acc[key] = AverageMeter()
        total_cls_loss[key] = 0.0
        correct_cnt[key] = 0
        total_cls_cnt[key] = torch.zeros(4)
        correct_cls_cnt[key] = torch.zeros(4)
    total_loss = 0.0
    model.eval()
    end = time.time()

    with torch.no_grad():
        for step, (sample_clip, recon_clip, step_label, recon_rate, motion_mask, recon_flags) in enumerate(val_loader):
            data_time.update(time.time() - end)

            clip_input = sample_clip.cuda()
            clip_label = recon_clip.cuda()
            step_label = step_label.cuda()
            recon_rate = recon_rate.cuda()
            motion_mask = motion_mask.cuda()
            recon_flags = recon_flags.cuda()

            clip_output, step_output, feat_output1 = model(clip_input)
            if args.mask_name == 'cam':
                fc8 = dict(model.named_parameters())['module.fc8_c5.weight'].detach()
                feat_mask = Cam_Mask(feat_output1.detach(), fc8, recon_clip, step_label, act_fun=args.mask_act)
            elif args.mask_name == 'patch':
                feat_mask = 0

            mask_lambda = Mask_lambda(epoch, lambda_str=args.mask_w_fun, max_epoch=args.epochs)
            mask = mask_lambda * feat_mask + (1 - mask_lambda) * motion_mask
            loss_recon = criterion_MSE(clip_output, clip_label, mask, recon_flags)
            loss_class = {}
            for key in encoder_cls_head_keys:
                loss_class[key] = criterion_CE(step_output[key], step_label)
            if encoder_cls_head_keys == ['c5', 'c4', 'c3', 'c2', 'c1']:
                loss = loss_recon + loss_class['c5'] * 0.1 + 0.1 * (loss_class['c4'] + loss_class['c3'] + loss_class['c2'] + loss_class['c1']) / 4
            elif encoder_cls_head_keys == ['c5', 'c4', 'c3', 'c2']:
                loss = loss_recon + loss_class['c5'] * 0.1 + 0.1 * (loss_class['c4'] + loss_class['c3'] + loss_class['c2']) / 3
            elif encoder_cls_head_keys == ['c5', 'c4', 'c3']:
                loss = loss_recon + loss_class['c5'] * 0.1 + 0.1 * (loss_class['c4'] + loss_class['c3']) / 2
            elif encoder_cls_head_keys == ['c5', 'c4']:
                loss = loss_recon + loss_class['c5'] * 0.1 + 0.1 * loss_class['c4']
            else:
                loss = loss_recon + loss_class['c5'] * 0.1

            batch_time.update(time.time() - end)
            end = time.time()

            losses_recon.update(loss_recon.item(), clip_input.size(0))
            losses.update(loss.item(), clip_input.size(0))
            for key in encoder_cls_head_keys:
                losses_class[key].update(loss_class[key].item(), clip_input.size(0))
                prec_class = accuracy(step_output[key].data, step_label, topk=(1,))[0]
                acc[key].update(prec_class.item(), clip_input.size(0))
                total_cls_loss[key] += loss_class[key].item()
                pts = torch.argmax(step_output[key], dim=1)
                correct_cnt[key] += torch.sum(step_label == pts).item()
                for i in range(step_label.size(0)):
                    total_cls_cnt[key][step_label[i]] += 1
                    if step_label[i] == pts[i]:
                        correct_cls_cnt[key][pts[i]] += 1

            if (step + 1) % params['display'] == 0:
                print('-----------------------------validation-------------------')
                p_str = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_loader))
                print(p_str)

                p_str = 'data_time:{data_time:.3f},batch time:{batch_time:.3f}'.format(data_time=data_time.val,
                                                                                       batch_time=batch_time.val)
                print(p_str)

                p_str = "loss:{loss:.5f} loss_recon:{loss_recon:.5f}".format(loss=losses.avg,
                                                                             loss_recon=losses_recon.avg)
                for key in encoder_cls_head_keys:
                    p_str += " loss_cls_{}:{:.5f}".format(key, losses_class[key].avg)
                print(p_str)

                p_str = ''
                for key in encoder_cls_head_keys:
                    p_str += ' acc_{}:{:.3f}'.format(key, acc[key].avg)
                print(p_str)

    for key in encoder_cls_head_keys:
        avg_cls_loss = total_cls_loss[key] / len(val_loader)
        avg_acc = correct_cnt[key] / len(val_loader.dataset)
        print('[VAL] loss_cls_{}: {:.3f}, acc_{}: {:.3f}'.format(key, avg_cls_loss, key, avg_acc))
        print(correct_cls_cnt[key])
        print(total_cls_cnt[key])
        print(correct_cls_cnt[key] / total_cls_cnt[key])

    avg_loss = losses.avg
    return avg_loss


# def load_pretrained_weights(ckpt_path):
#     adjusted_weights = {};
#     pretrained_weights = torch.load(ckpt_path, map_location='cpu');
#     for name, params in pretrained_weights.items():
#         if "module" in name:
#             name = name[name.find('.') + 1:]
#         adjusted_weights[name] = params;
#     return adjusted_weights;

def load_pretrained_weights(ckpt_path):
    adjusted_weights = {}
    pretrained_weights = torch.load(ckpt_path, map_location='cpu')
    for name, params in pretrained_weights.items():
        if "module" in name:
            name = 'base_network.' + name[name.find('.') + 1:]
        if "linear" not in name:
            adjusted_weights[name] = params
    return adjusted_weights


def parse_args():
    parser = argparse.ArgumentParser(description='Video Clip Restruction and Order Prediction')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
    parser.add_argument('--model_name', type=str, default='c3d', help='model name')
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    parser.add_argument('--ma_mode', type=str, default='DPAU', help='motion attention mode')
    parser.add_argument('--mask_name', type=str, default='cam', help='mask name')
    parser.add_argument('--mask_w_fun', type=str, default='exp', help='mask weight function name')
    parser.add_argument('--enc_head', type=str, default='c5_c4_c3_c2', help='encoder cls heads')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    save_path = params['save_path_base'] + "train_predict_{}_".format(args.exp_name) + params['data']
    model_save_dir = os.path.join(save_path, time.strftime('%m-%d-%H-%M'))
    writer = SummaryWriter(model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    log_file = os.path.join(model_save_dir, 'log.txt')
    sys.stdout = Logger(log_file)
    print(vars(args))

    if args.model_name == 'c3d':
        print(args.model_name)
        model = c3d.C3D_Hed(with_classifier=False)
    elif args.model_name == 'r3d':
        print(args.model_name)
        model = r3d.R3DNet_Hed((1, 1, 1, 1), with_classifier=False)
    elif args.model_name == 'r21d':
        print(args.model_name)
        model = r21d.R2Plus1DNet_Hed((1, 1, 1, 1), with_classifier=False)

    model = sscn.SSCN_OneClip(args.model_name, base_network=model, with_classifier=True, num_classes=4, with_ClsEncoder=args.enc_head.split('_'))
    print(model)
    if ckpt:
        weight = load_pretrained_weights(ckpt)
        model.load_state_dict(weight, strict=False)
    # train

    image_augmentation = None
    video_augmentation = transforms.Compose([
        video_transforms.ToPILImage(),
        video_transforms.Resize((128, 171)),
        video_transforms.RandomCrop(112),
        video_transforms.ToTensor()
    ])

    train_dataset = PredictDataset(params['dataset'], mode="train", dataset=params['data'],
                                   video_transforms=video_augmentation, image_transforms=image_augmentation, args=args)
    if params['data'] == 'kinetics-400':
        val_dataset = PredictDataset(params['dataset'], mode='val', dataset=params['data'],
                                     video_transforms=video_augmentation, image_transforms=image_augmentation,
                                     args=args)

    elif params['data'] == 'UCF-101':
        val_size = 800
        train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset) - val_size, val_size))
    elif params['data'] == 'hmdb':
        val_size = 400
        train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset) - val_size, val_size))

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                              num_workers=params['num_workers'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=True,
                            num_workers=params['num_workers'], drop_last=True)
    if multi_gpu == 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    criterion_CE = nn.CrossEntropyLoss().cuda()
    criterion_MSE = Motion_MSEloss_NFGT().cuda()

    model_params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'fc8' in key:
                print(key)
                model_params += [{'params': [value], 'lr': 10 * learning_rate}]
            else:
                model_params += [{'params': [value], 'lr': learning_rate}]
    optimizer = optim.SGD(model_params, momentum=params['momentum'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=50, factor=0.1)

    prev_best_val_loss = 100
    prev_best_loss_model_path = None
    for epoch in tqdm(range(start_epoch, start_epoch + args.epochs)):
        train(train_loader, model, criterion_MSE, criterion_CE, optimizer, epoch, writer, args=args)
        val_loss = validation(val_loader, model, criterion_MSE, criterion_CE, optimizer, epoch, args=args)
        if val_loss < prev_best_val_loss:
            model_path = os.path.join(model_save_dir, 'best_model_{}.pth.tar'.format(epoch))
            torch.save(model.state_dict(), model_path)
            prev_best_val_loss = val_loss
            if prev_best_loss_model_path:
                os.remove(prev_best_loss_model_path)
            prev_best_loss_model_path = model_path
        scheduler.step(val_loss)

        if epoch % 20 == 0:
            checkpoints = os.path.join(model_save_dir, 'model_{}.pth.tar'.format(epoch))
            torch.save(model.state_dict(), checkpoints)
            print("save_to:", checkpoints)


if __name__ == '__main__':
    seed = 632
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()

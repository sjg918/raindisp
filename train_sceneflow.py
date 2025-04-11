from __future__ import print_function, division
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
import datetime
#from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss_train, model_loss_test
from utils import *
from torch.utils.data import DataLoader
import gc

cudnn.benchmark = True
# nohup python train_sceneflow.py 1> /dev/null 2>&1 &

parser = argparse.ArgumentParser(description='Point-Wise Stereo Matching')
parser.add_argument('--model', default='PW_Stereo_v3', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--window_size', type=int, default=7, help='block window size')
parser.add_argument('--num_points', type=int, default=1024, help='number of points')

parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/home/jnu-ie/Dataset/sceneflow/", help='data path')
parser.add_argument('--trainlist', default='./filenames/sceneflow_train.txt', help='training list')
parser.add_argument('--testlist',default='./filenames/sceneflow_test.txt', help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default="8,11,13,15:2", help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default='', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', default='', help='load the weights from a specific checkpoint')

parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
#logger = SummaryWriter(args.logdir)
print("creating new summary file")
if os.path.exists(args.logdir):
    pass
else:
    os.makedirs(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp, args.window_size, args.num_points)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()
#optimizer = optim.AdamW(model.parameters(), lr=args.lr)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict) 
    # model.load_state_dict(state_dict['model'])
    model.load_state_dict(model_dict)
print("start at epoch {}".format(start_epoch))


def train():
    with open(args.logdir + 'log.txt', 'w') as writer:
       writer.write("-start-\t")
       writer.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
       writer.write('\n\n')

    bestepoch = 0
    error = 100
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)
        runningloss = []
        timelist = []

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            #start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = train_sample(sample, compute_metrics=do_summary)
            
            # if do_summary:
            #     save_scalars(logger, 'train', scalar_outputs, global_step)
            #     # save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs
            runningloss.append(loss)
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss))
        with open(args.logdir + 'log.txt', 'a') as writer:
            writer.write("{}/{} losses: {:.5f} \t".format(epoch_idx, args.epochs, sum(runningloss) / len(runningloss)))
            writer.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            writer.write('\n')
        
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        #bestepoch = 0
        #error = 100
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx

            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = test_sample(sample, timelist, compute_metrics=do_summary)

            # if do_summary:
            #     save_scalars(logger, 'test', scalar_outputs, global_step)
            #     # save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoader), loss))
        avg_test_scalars = avg_test_scalars.mean()
        nowerror = avg_test_scalars["D1"]
        if  nowerror < error :
            bestepoch = epoch_idx
            error = avg_test_scalars["D1"]
        #save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        with open(args.logdir + 'log.txt', 'a') as writer:
            writer.write('avg test scalars :: (loss {:.5f}) (D1 {:.5f}) (EPE {:.5f}) (meantime {:.5f})\n'.format(avg_test_scalars['loss'], avg_test_scalars['D1'], avg_test_scalars['EPE'], sum(timelist)/len(timelist)))
            writer.write('MAX epoch %d total test error = %.5f\t' % (bestepoch, error))
            writer.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            writer.write('\n')
        gc.collect()

        continue

    with open(args.logdir + 'log.txt', 'a') as writer:
        writer.write('MAX epoch %d total test error = %.5f \n' % (bestepoch, error))
        writer.write('-end-\t')
        writer.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()
    imgL, imgR, disp_gt, points = sample['left'], sample['right'], sample['disparity'], sample['points']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    points = points.cuda()
    optimizer.zero_grad()

    if 'onlyedge' in args.logdir:
        loss, pred, gt, mask = model(imgL, imgR, disp_gt, points)
    elif 'halfedge' in args.logdir:
        loss, pred, gt, mask = model(imgL, imgR, disp_gt, points[:, :32, :])
    else:
        loss, pred, gt, mask = model(imgL, imgR, disp_gt)
    loss = loss.sum()

    scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            # image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests_final]
            scalar_outputs["EPE"] = EPE_metric(pred, gt, mask)
            scalar_outputs["D1"] = D1_metric(pred, gt, mask)
            # scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests_final]
            # scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests_final]
            # scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests_final]
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs)


def pooling_gt_disparity(disp, points, img_w, img_h):
    img_w = img_w - 1
    img_h = img_h - 1
    points[:, :, 0] = points[:, :, 0] / img_w * 2. - 1.
    points[:, :, 1] = points[:, :, 1] / img_h * 2. - 1.
    points = points.unsqueeze(2)
    gt_disp = F.grid_sample(disp.unsqueeze(1), points, align_corners=True)
    gt_disp = gt_disp.squeeze(-1).squeeze(1)
    return gt_disp


# test one sample
@make_nograd_func
def test_sample(sample, timelist, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    _, _, H, W = imgL.shape
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    
    torch.cuda.synchronize()
    start_time = time.time()
    pred, points, _ = model(imgL, imgR)
    torch.cuda.synchronize()
    timelist.append(time.time() - start_time)

    gt = pooling_gt_disparity(disp_gt, points, W, H)

    mask = (gt < args.maxdisp) & (gt > 0)
    if mask.sum() == 0:
        loss = F.smooth_l1_loss(pred, pred, reduction='mean')
    else:
        loss = F.smooth_l1_loss(pred[mask], gt[mask], reduction='mean')

    loss = loss.sum()
    scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = D1_metric(pred, gt, mask)
    scalar_outputs["EPE"] = EPE_metric(pred, gt, mask)
    scalar_outputs["Thres1"] = Thres_metric(pred, gt, mask, 1.0) 
    scalar_outputs["Thres2"] = Thres_metric(pred, gt, mask, 2.0)
    scalar_outputs["Thres3"] = Thres_metric(pred, gt, mask, 3.0)

    # if compute_metrics:
    #     image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs)

if __name__ == '__main__':
    train()

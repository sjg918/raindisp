
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import argparse
import time
from PIL import Image
import cv2

from datasets.data_io import get_transform
from models import __models__

parser = argparse.ArgumentParser(description='Point-Wise Stereo Matching')
parser.add_argument('--model', default='PW_Stereo_v3', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--window_size', type=int, default=7, help='block window size')
parser.add_argument('--num_points', type=int, default=64, help='number of points')
parser.add_argument('--loadckpt', default='./checkpoints/PW_Stereo_v2-kitti-halfedge/checkpoint_000484.ckpt', help='')

args = parser.parse_args()

    
def D1_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

def EPE_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    return F.l1_loss(D_est, D_gt, size_average=True)

def pooling_gt_disparity(disp, points, img_w, img_h):
    img_w = img_w - 1
    img_h = img_h - 1
    points_ = torch.zeros_like(points)
    points_[:, :, 0] = points[:, :, 0] / img_w * 2. - 1.
    points_[:, :, 1] = points[:, :, 1] / img_h * 2. - 1.
    points_ = points_.unsqueeze(2)
    gt_disp = F.grid_sample(disp.unsqueeze(1), points_, align_corners=True)
    gt_disp = gt_disp.squeeze(-1).squeeze(1)
    return gt_disp

def mainf():
    model = __models__[args.model](args.maxdisp, args.window_size, args.num_points)
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)

    model.eval()
    timelist = []

    imglist = os.listdir('./images/day-Limg/')
    D1, EPE = 0, 0
    with torch.no_grad():
        for cnt, image_name in enumerate(imglist):

            Limg = Image.open('./images/day-Limg/' + image_name).convert('RGB')
            Rimg = Image.open('./images/day-Rimg/' + image_name).convert('RGB')
            anno = np.load('./images/day-anno/' + image_name.split('.')[0] + '.npy')

            processed = get_transform()
            Limg = processed(Limg).numpy()
            Rimg = processed(Rimg).numpy()

            Limg = torch.from_numpy(Limg).cuda().unsqueeze(0)
            Rimg = torch.from_numpy(Rimg).cuda().unsqueeze(0)
            anno = anno.astype(np.float32)
            anno = torch.from_numpy(anno).cuda().unsqueeze(0)

            torch.cuda.synchronize()
            start_time = time.time()
            pred, _ = model(Limg, Rimg, None, anno[:, :, :2])
            torch.cuda.synchronize()
            if cnt == 0:
                pass
            else:
                timelist.append(time.time() - start_time)
            cnt = cnt + 1
            
            gt = anno[:, :, 2]
            mask = (gt < args.maxdisp) & (gt > 0)
            D1 = D1 + D1_metric(pred, gt, mask).item()
            EPE = EPE + EPE_metric(pred, gt, mask).item()         
    
            continue
    print('D1, EPE : ', D1 / len(imglist), EPE / len(imglist))
    print('mean Time : ', sum(timelist) / len(timelist))

    return 0
    


if __name__ == '__main__':
    mainf()

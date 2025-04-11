from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data

import torch.nn.functional as F
from .submodule import *
import math


class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class FEN(SubModule):
    def __init__(self):
        super(FEN, self).__init__()
        self.conv1 = BasicConv(3, 32, kernel_size=7, padding=3, stride=1)
        self.conv2 = BasicConv(32, 32, kernel_size=1, padding=0, stride=1, bn=False, relu=False)

        self.conv3 = BasicConv(3, 32, kernel_size=7, padding=3, stride=1)
        self.conv4 = BasicConv(32, 32, kernel_size=1, padding=0, stride=1, bn=False, relu=False)

        self.conv5 = BasicConv(3, 32, kernel_size=7, padding=3, stride=1)
        self.conv6 = BasicConv(32, 32, kernel_size=1, padding=0, stride=1, bn=False, relu=False)

        self.conv7 = BasicConv(3, 32, kernel_size=7, padding=3, stride=1)
        self.conv8 = BasicConv(32, 32, kernel_size=1, padding=0, stride=1, bn=False, relu=False)

        self.conv9 = BasicConv(3, 32, kernel_size=7, padding=3, stride=1)
        self.conv10 = BasicConv(32, 32, kernel_size=1, padding=0, stride=1, bn=False, relu=False)

        self.conv11 = BasicConv(3, 32, kernel_size=7, padding=3, stride=1)
        self.conv12 = BasicConv(32, 32, kernel_size=1, padding=0, stride=1, bn=False, relu=False)

        self.weight_init()

    def forward(self, x):
        B, C, H, W = x.shape
        x_0 = x
        x_2 = F.interpolate(x, (H//2, W//2), mode='bilinear', align_corners=True)
        x_4 = F.interpolate(x, (H//4, W//4), mode='bilinear', align_corners=True)
        x_8 = F.interpolate(x, (H//8, W//8), mode='bilinear', align_corners=True)
        x_16 = F.interpolate(x, (H//16, W//16), mode='bilinear', align_corners=True)
        x_32 = F.interpolate(x, (H//32, W//32), mode='bilinear', align_corners=True)

        x1 = self.conv1(x_0)
        x2 = self.conv2(x1)

        x3 = self.conv3(x_2)
        x4 = self.conv4(x3)

        x5 = self.conv5(x_4)
        x6 = self.conv6(x5)

        x7 = self.conv7(x_8)
        x8 = self.conv8(x7)

        x9 = self.conv9(x_16)
        x10 = self.conv10(x9)

        x11 = self.conv11(x_32)
        x12 = self.conv12(x11)

        x0_list = []
        x2_list = []
        x4_list = []
        x8_list = []
        x16_list = []
        x32_list = []

        for i in range(8):
            x0_list.append(x2[:, i*4:(i+1)*4, :, :])
            x2_list.append(x4[:, i*4:(i+1)*4, :, :])
            x4_list.append(x6[:, i*4:(i+1)*4, :, :])
            x8_list.append(x8[:, i*4:(i+1)*4, :, :])
            x16_list.append(x10[:, i*4:(i+1)*4, :, :])
            x32_list.append(x12[:, i*4:(i+1)*4, :, :])
            continue

        # x2 = [x2[:, i*4:(i+1)*4, :, :] for i in range(8)]
        # x4 = [x4[:, i*4:(i+1)*4, :, :] for i in range(8)]
        # x6 = [x6[:, i*4:(i+1)*4, :, :] for i in range(8)]
        # x8 = [x8[:, i*4:(i+1)*4, :, :] for i in range(8)]
        # x10 = [x10[:, i*4:(i+1)*4, :, :] for i in range(8)]
        # x12 = [x12[:, i*4:(i+1)*4, :, :] for i in range(8)]

        #return x2 + x4 + x6 + x8 + x10 + x12
        return x0_list + x2_list + x4_list + x8_list + x16_list + x32_list


class PointWiseBlockMaching(nn.Module):
    def __init__(self, maxdisp=192, subscale=(1,4), window_size=5):
        super().__init__()
        self.window_size = window_size
        self.subscale = subscale
        self.maxdisp = maxdisp

        for scale in subscale:
            disp = torch.arange(0, maxdisp, step=scale, dtype=torch.float32)
            disp = disp.repeat_interleave(window_size*window_size).view(-1, window_size*window_size)
            self.register_buffer(name='disp_' + str(scale), tensor=disp)

            xwindow = torch.arange(0, window_size * scale, step=scale, dtype=torch.float32) - (window_size//2 * scale)
            xwindow = xwindow.repeat(window_size).view(window_size, window_size)
            self.register_buffer(name='xwindow_' + str(scale), tensor=xwindow)
            ywindow = torch.arange(0, window_size * scale, step=scale, dtype=torch.float32) - (window_size//2 * scale)
            ywindow = ywindow.repeat(window_size).view(window_size, window_size).permute(1, 0).contiguous()
            self.register_buffer(name='ywindow_' + str(scale), tensor=ywindow)


    def __call__(self, left_list, right_list, points, img_w, img_h, disp_real=None, reverse=False):
        # points -> start at 0. not 1.
        img_w = img_w - 1
        img_h = img_h - 1
        B, _, _, _ = left_list[0].shape
        B, N, _ = points.shape

        if reverse:
            disp_real_ = disp_real.view(B, N, 1).repeat(1, 1, self.window_size*self.window_size)

        left_points_list = []
        right_points_list = []
        for i, scale in enumerate(self.subscale):
            # batch, num of points, window*window, (y,x)
            left_points = points.repeat(1, 1, self.window_size*self.window_size).view(B, N, self.window_size*self.window_size, 2)
            left_points[:, :, :, 0] = left_points[:, :, :, 0] + getattr(self, 'xwindow_' + str(scale)).view(1, 1, self.window_size*self.window_size).to(device=points.device)
            if reverse:
                left_points[:, :, :, 0] = left_points[:, :, :, 0] - disp_real_ / scale
            left_points[:, :, :, 1] = left_points[:, :, :, 1] + getattr(self, 'ywindow_' + str(scale)).view(1, 1, self.window_size*self.window_size).to(device=points.device)

            # batch, num of points, maxdisp*window*window, (y,x)
            right_points = left_points.view(B, N, 1, self.window_size*self.window_size, 2).repeat(
                    1, 1, self.maxdisp//scale, 1, 1)
            if reverse:
                right_points[:, :, :, :, 0] = right_points[:, :, :, :, 0] + getattr(self, 'disp_' + str(scale)).view(
                    1, 1, self.maxdisp//scale, self.window_size*self.window_size).to(device=points.device)
            else:
                right_points[:, :, :, :, 0] = right_points[:, :, :, :, 0] - getattr(self, 'disp_' + str(scale)).view(
                    1, 1, self.maxdisp//scale, self.window_size*self.window_size).to(device=points.device)
            right_points = right_points.view(B, N, self.maxdisp//scale*self.window_size*self.window_size, 2)

            # convert to grid
            left_points[:, :, :, 0] = left_points[:, :, :, 0] / img_w * 2. - 1.
            left_points[:, :, :, 1] = left_points[:, :, :, 1] / img_h * 2. - 1.
            right_points[:, :, :, 0] = right_points[:, :, :, 0] / img_w * 2. - 1.
            right_points[:, :, :, 1] = right_points[:, :, :, 1] / img_h * 2. - 1.

            left_points_list.append(left_points)
            right_points_list.append(right_points)
            continue

        # cost
        cost_list = []
        for i in range(len(left_list)):
            left_feature = left_list[i]
            right_feature = right_list[i]
            _, C, _, _ = left_feature.shape 

            left_points = left_points_list[i//8]
            right_points = right_points_list[i//8]
            scale = self.subscale[i//8]

            # batch, num of points, window, window, channels
            left_feature = F.grid_sample(left_feature, left_points, align_corners=True)
            left_feature = left_feature.view(B, C, N, self.window_size, self.window_size).permute(0, 2, 3, 4, 1).contiguous()

            # batch, num of points, maxdisp, window, window, channels
            right_feature = F.grid_sample(right_feature, right_points, align_corners=True)
            right_feature = right_feature.view(B, C, N, self.maxdisp//scale, self.window_size, self.window_size).permute(0, 2, 3, 4, 5, 1).contiguous()

            #
            left_feature = left_feature.view(B, N, 1, self.window_size, self.window_size, C).repeat(1, 1, self.maxdisp//scale, 1, 1, 1)
            cost = (left_feature - right_feature).abs().view(B, N, self.maxdisp//scale, -1).mean(-1, keepdim=True)
            cost = 1 - torch.exp(-1.0 * cost)

            if scale != 1:
                cost = F.interpolate(cost, (self.maxdisp, 1), mode='bilinear', align_corners=True)

            cost_list.append(cost)
            continue

        # batch, num of points, maxdisp, c
        cost_list = torch.cat((cost_list), dim=-1)
        cost_list = cost_list.view(B, N, self.maxdisp, len(self.subscale), -1).sum(dim=3, keepdim=False)
        return cost_list


class CostAggregation(SubModule):
    def __init__(self):
        super(CostAggregation, self).__init__()
        self.conv1 = BasicConv(8, 8, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.conv2 = BasicConv(8, 8, kernel_size=(3, 1), padding=(1, 0), stride=1)

        self.conv3 = BasicConv(8, 8, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.conv4 = BasicConv(8, 8, kernel_size=(3, 1), padding=(1, 0), stride=1)

        self.conv5 = BasicConv(8, 8, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.conv6 = BasicConv(8, 8, kernel_size=(3, 1), padding=(1, 0), stride=1)

        self.conv7 = BasicConv(8, 8, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.conv8 = BasicConv(8, 8, kernel_size=(3, 1), padding=(1, 0), stride=1)

        # self.conv9 = BasicConv(8, 8, kernel_size=(3, 1), padding=(1, 0), stride=1)
        # self.conv10 = BasicConv(8, 8, kernel_size=(3, 1), padding=(1, 0), stride=1)

        self.conv11 = BasicConv(8, 1, kernel_size=(1, 1), padding=(0, 0), stride=1, bn=False, relu=False)
        self.weight_init()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1) + x

        x3 = self.conv3(x2)
        x4 = self.conv4(x3) + x2

        x5 = self.conv5(x4)
        x6 = self.conv6(x5) + x4

        x7 = self.conv7(x6)
        x8 = self.conv8(x7) + x6

        # x9 = self.conv9(x8)
        # x10 = self.conv10(x9) + x9

        x11 = self.conv11(x8)
        return x11


class DisparityRegression(SubModule):
    def __init__(self, maxdisp, k=2):
        super().__init__()
        self.maxdisp = maxdisp
        self.k = k

    def forward(self, cost):
        disp_samples = torch.arange(0, self.maxdisp, dtype=cost.dtype, device=cost.device)
        disp_samples = disp_samples.view(1, 1, self.maxdisp).repeat(cost.shape[0], cost.shape[1], 1)

        _, ind = cost.sort(-1, True)
        pool_ind = ind[:, :, :self.k]
        cost = torch.gather(cost, -1, pool_ind)
        prob = F.softmax(cost, -1)
        disp_samples = torch.gather(disp_samples, -1, pool_ind)
        pred = torch.sum(disp_samples * prob, dim=-1, keepdim=False)
        return pred


class PointWiseStereo_v3(nn.Module):
    def __init__(self, maxdisp, window_size, num_points):
        super(PointWiseStereo_v3, self).__init__()
        self.maxdisp = maxdisp
        self.num_points = num_points
        self.feature = FEN()
        self.pwbm = PointWiseBlockMaching(maxdisp, (1,2,4,8,16,32), window_size)
        self.aggregation = CostAggregation()
        self.regression = DisparityRegression(maxdisp, k=2)
        self.check_lr = True

    def pooling_gt_disparity(self, disp, points, img_w, img_h):
        img_w = img_w - 1
        img_h = img_h - 1
        points[:, :, 0] = points[:, :, 0] / img_w * 2. - 1.
        points[:, :, 1] = points[:, :, 1] / img_h * 2. - 1.
        points = points.unsqueeze(2)
        gt_disp = F.grid_sample(disp.unsqueeze(1), points, align_corners=True)
        gt_disp = gt_disp.squeeze(-1).squeeze(1)
        return gt_disp

    def forward(self, left, right, disp=None, points=None):
        B, C, H, W = left.shape

        # generate random poitns
        if points is None:
            points_w = torch.randint(0, W, (B, self.num_points, 1), dtype=torch.float32, device=left.device)
            points_h = torch.randint(0, H, (B, self.num_points, 1), dtype=torch.float32, device=left.device)
            points = torch.cat((points_w, points_h), dim=-1)
            num_points = self.num_points
        elif points.shape[1] < self.num_points and self.training:
            n = points.shape[1]
            points_w = torch.randint(0, W, (B, self.num_points - n, 1), dtype=torch.float32, device=left.device)
            points_h = torch.randint(0, H, (B, self.num_points - n, 1), dtype=torch.float32, device=left.device)
            points_new = torch.cat((points_w, points_h), dim=-1)
            points = torch.cat((points, points_new), dim=1)
            num_points = self.num_points
        else:
            _, num_points, _ = points.shape

        # feature extraction
        left = self.feature(left)
        right = self.feature(right)
        
        # point wise block matching
        cost = self.pwbm(left, right, points, W, H)
        # batch, num of points, maxdisp, c -> batch * num of points, c, maxdisp, 1
        cost = cost.view(B * num_points, self.maxdisp, -1).permute(
            0, 2, 1).contiguous().unsqueeze(-1)

        # aggregation
        cost = self.aggregation(cost)
        # batch * num of points, 1, maxdisp, 1 -> batch * num of points, maxdisp
        cost = cost.squeeze(-1).squeeze(1).view(B, num_points, self.maxdisp)

        # regression
        pred = self.regression(cost)
        #return True, True
        if self.training:
            gt = self.pooling_gt_disparity(disp, points, W, H)
            mask = (gt < self.maxdisp) & (gt > 0)
            if mask.sum() == 0:
                loss = F.smooth_l1_loss(pred, pred, reduction='mean')
            else:
                loss = F.smooth_l1_loss(pred[mask], gt[mask], reduction='mean')

            return loss, pred, gt, mask

        invaild_mask = None
        if self.check_lr:
            reverse_cost = self.pwbm(right, left, points, W, H, disp_real=pred, reverse=True)
            reverse_cost = reverse_cost.view(B * num_points, self.maxdisp, -1).permute(
                0, 2, 1).contiguous().unsqueeze(-1)
            reverse_cost = self.aggregation(reverse_cost)
            reverse_cost = reverse_cost.squeeze(-1).squeeze(1).view(B, num_points, self.maxdisp)
            reverse_pred = self.regression(reverse_cost)

            invaild_mask1 = (pred - reverse_pred).abs() > 3
            #invaild_mask2 = torch.maximum(pred / reverse_pred, reverse_pred / pred) > 1.05
            #invaild_mask = torch.logical_or(invaild_mask1, invaild_mask2)
            invaild_mask = invaild_mask1
            if num_points - invaild_mask.sum() < 5:
                pass
            else:
                vaild_mask = torch.logical_not(invaild_mask)
                vaild_points_y = points[:, :, 1][vaild_mask].view(B, -1)
                vaild_disp = pred[vaild_mask].view(B, -1)
                _, N_vaild = vaild_points_y.shape

                # 2 fit
                A = torch.zeros((B, N_vaild, 2), dtype=torch.float32, device=points.device)
                A[:, :, 0] = vaild_points_y
                A[:, :, 1] = 1
                X = torch.linalg.lstsq(A, vaild_disp)[0]

                A = torch.zeros((B, num_points, 2), dtype=torch.float32, device=points.device)
                A[:, :, 0] = points[:, :, 1]
                A[:, :, 1] = 1
                fit_disp = (A @ X.unsqueeze(-1)).squeeze(-1)

                # 3 fit
                # A = torch.zeros((B, N_vaild, 3), dtype=torch.float32, device=points.device)
                # A[:, :, 0] = vaild_points_y * vaild_points_y
                # A[:, :, 1] = vaild_points_y
                # A[:, :, 2] = 1
                # X = torch.linalg.lstsq(A, vaild_disp)[0]

                # A = torch.zeros((B, num_points, 3), dtype=torch.float32, device=points.device)
                # A[:, :, 0] = points[:, :, 1] * points[:, :, 1]
                # A[:, :, 1] = points[:, :, 1]
                # A[:, :, 2] = 1
                # fit_disp = (A @ X.unsqueeze(-1)).squeeze(-1)
                
                pred[invaild_mask] = fit_disp[invaild_mask]
                #pred[invaild_mask] = -1
            

        return pred, torch.logical_not(invaild_mask) if invaild_mask is not None else None

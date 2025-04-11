from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data

import torch.nn.functional as F
from .submodule import *
import math


# refer : https://github.com/Tianxiaomo/pytorch-YOLOv4
class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, depthwise=False, dilation=1, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if depthwise and dilation == 1:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation=dilation, groups=in_channels, bias=bias))
        elif dilation == 1:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation=dilation, bias=bias))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation*pad, dilation=dilation, bias=bias))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


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
        self.conv1 = Conv_Bn_Activation(3, 24, 7, 1, 'leaky', dilation=1, bn=True, bias=False)
        self.conv2 = Conv_Bn_Activation(3, 24, 7, 1, 'leaky', dilation=2, bn=True, bias=False)
        self.conv3 = Conv_Bn_Activation(3, 24, 7, 1, 'leaky', dilation=4, bn=True, bias=False)
        self.conv4 = Conv_Bn_Activation(3, 24, 7, 1, 'leaky', dilation=8, bn=True, bias=False)
        self.conv5 = Conv_Bn_Activation(3, 24, 7, 1, 'leaky', dilation=16, bn=True, bias=False)
        self.conv6 = Conv_Bn_Activation(3, 24, 7, 1, 'leaky', dilation=32, bn=True, bias=False)

        self.convc = Conv_Bn_Activation(6, 8, 1, 1, 'leaky', dilation=32, bn=True, bias=False)

        self.conv7 = Conv_Bn_Activation(32, 32, 7, 1, 'linear', dilation=1, bn=False, bias=True)
        self.conv8 = Conv_Bn_Activation(32, 32, 7, 1, 'linear', dilation=1, bn=False, bias=True)
        self.conv9 = Conv_Bn_Activation(32, 32, 7, 1, 'linear', dilation=1, bn=False, bias=True)
        self.conv10 = Conv_Bn_Activation(32, 32, 7, 1, 'linear', dilation=1, bn=False, bias=True)
        self.conv11 = Conv_Bn_Activation(32, 32, 7, 1, 'linear', dilation=1, bn=False, bias=True)
        self.conv12 = Conv_Bn_Activation(32, 32, 7, 1, 'linear', dilation=1, bn=False, bias=True)
        self.weight_init()

    def forward(self, x):
        B, C, H, W = x.shape
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        x6 = self.conv6(x)

        xc = torch.cat(
            (x1.mean(dim=1, keepdim=True),
             x2.mean(dim=1, keepdim=True),
             x3.mean(dim=1, keepdim=True),
             x4.mean(dim=1, keepdim=True),
             x5.mean(dim=1, keepdim=True),
             x6.mean(dim=1, keepdim=True)), dim=1)
        xc = self.convc(xc)

        x1 = self.conv7(torch.cat((x1, xc), dim=1))
        x2 = self.conv8(torch.cat((x2, xc), dim=1))
        x3 = self.conv9(torch.cat((x3, xc), dim=1))
        x4 = self.conv10(torch.cat((x4, xc), dim=1))
        x5 = self.conv11(torch.cat((x5, xc), dim=1))
        x6 = self.conv12(torch.cat((x6, xc), dim=1))

        xc = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        
        return xc


class PointWiseBlockMaching(nn.Module):
    def __init__(self, maxdisp=192, window_size=5):
        super().__init__()
        self.window_size = window_size
        scale = 1
        self.maxdisp = maxdisp

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
        B, _, _, _ = left_list.shape
        B, N, _ = points.shape
        G = 48

        if reverse:
            disp_real_ = disp_real.view(B, N, 1).repeat(1, 1, self.window_size*self.window_size)


        scale = 1
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

        # cost
        left_feature = left_list
        right_feature = right_list
        _, C, _, _ = left_feature.shape 

        # batch, num of points, group, window, window, channels
        left_feature = F.grid_sample(left_feature, left_points, align_corners=True)
        left_feature = left_feature.view(B, C, N, self.window_size, self.window_size)
        left_feature = left_feature.view(B, G, C//G, N, self.window_size, self.window_size).permute(0, 3, 1, 4, 5, 2).contiguous()

        # batch, num of points, maxdisp, group, window, window, channels
        right_feature = F.grid_sample(right_feature, right_points, align_corners=True)
        right_feature = right_feature.view(B, C, N, self.maxdisp//scale, self.window_size, self.window_size)
        right_feature = right_feature.view(B, G, C//G, N, self.maxdisp//scale, self.window_size, self.window_size).permute(0, 3, 4, 1, 5, 6, 2).contiguous()

        # batch, num of points, maxdisp, group, window, window, channels
        left_feature = left_feature.view(B, N, 1, G, self.window_size, self.window_size, C//G).repeat(1, 1, self.maxdisp//scale, 1, 1, 1, 1)
        cost = (left_feature - right_feature).abs().view(B, N, self.maxdisp//scale, G, -1).mean(-1, keepdim=True)
        cost = 1 - torch.exp(-1.0 * cost)

        # batch, num of points, maxdisp, c=(group)
        # cost_list = torch.cat((cost_list), dim=-1)
        # cost_list = cost_list.view(B, N, self.maxdisp, len(self.subscale), -1).sum(dim=3, keepdim=False)
        #print(cost.shape, cost.shape)
        return cost


class CostAggregation(SubModule):
    def __init__(self):
        super(CostAggregation, self).__init__()
        self.conv1 = BasicConv(48, 48, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.conv2 = BasicConv(48, 48, kernel_size=(3, 1), padding=(1, 0), stride=1)

        self.conv3 = BasicConv(48, 48, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.conv4 = BasicConv(48, 48, kernel_size=(3, 1), padding=(1, 0), stride=1)

        self.conv5 = BasicConv(48, 48, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.conv6 = BasicConv(48, 48, kernel_size=(3, 1), padding=(1, 0), stride=1)

        self.conv7 = BasicConv(48, 48, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.conv8 = BasicConv(48, 48, kernel_size=(3, 1), padding=(1, 0), stride=1)

        # self.conv9 = BasicConv(8, 8, kernel_size=(3, 1), padding=(1, 0), stride=1)
        # self.conv10 = BasicConv(8, 8, kernel_size=(3, 1), padding=(1, 0), stride=1)

        self.conv11 = BasicConv(48, 1, kernel_size=(1, 1), padding=(0, 0), stride=1, bn=False, relu=False)
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
        x11 = -1 * x11
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


class UL_Stereo_v1(nn.Module):
    def __init__(self, maxdisp, window_size, num_points):
        super(UL_Stereo_v1, self).__init__()
        self.maxdisp = maxdisp
        self.num_points = num_points
        self.feature = FEN()
        self.pwbm = PointWiseBlockMaching(maxdisp, window_size=1)
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
        #print(points.shape)
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
        #print(points.shape)
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

        if self.training:
            gt = self.pooling_gt_disparity(disp, points, W, H)
            mask = (gt < self.maxdisp) & (gt > 0)
            if mask.sum() == 0:
                loss = F.smooth_l1_loss(pred, pred, reduction='mean')
            else:
                loss = F.smooth_l1_loss(pred[mask], gt[mask], reduction='mean')

            return loss, pred, gt, mask

        if self.check_lr and self.training == False:
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
            return pred, points, torch.logical_not(invaild_mask)
        else:
            return pred, points, None

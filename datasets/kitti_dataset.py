import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
from . import flow_transforms
import torchvision
import cv2
import copy

import torch

class KITTIDataset(Dataset):
    def __init__(self, datapath_12, datapath_15, list_filename, training):
        self.datapath_15 = datapath_15
        self.datapath_12 = datapath_12
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.num_points = 1024
        if self.training:
            assert self.disp_filenames is not None

    def RGB2EdgePoints(self, img, thr=30):
        cv2limg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2sobel_x = cv2.Sobel(cv2limg, cv2.CV_32F, 1, 0, 3)
        cv2sobel_y = cv2.Sobel(cv2limg, cv2.CV_32F, 1, 0, 3) # we don't need y direction edges
        cv_sobel = cv2.magnitude(cv2sobel_x, cv2sobel_y)
        cv_sobel = np.clip(cv_sobel, 0, 255)
        cv_sobel = cv2.medianBlur(cv_sobel, 3)
        mask = (cv_sobel > thr)

        if np.sum(mask) == 0:
            return None

        H, W, _ = img.shape
        xcoord = np.arange(0, W)
        ycoord = np.arange(0, H)
        xcoord,ycoord = np.meshgrid(xcoord,ycoord)
        xcoord = xcoord[mask]
        ycoord = ycoord[mask]
        points = np.column_stack((xcoord, ycoord))
        return points

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):

        left_name = self.left_filenames[index].split('/')[1]
        if left_name.startswith('image'):
            self.datapath = self.datapath_15
        else:
            self.datapath = self.datapath_12
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))



        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None

        if self.training:
            th, tw = 256, 512
            random_brightness = np.random.uniform(0.5, 2.0, 2)
            random_gamma = np.random.uniform(0.8, 1.2, 2)
            random_contrast = np.random.uniform(0.8, 1.2, 2)
            random_saturation = np.random.uniform(0, 1.4, 2)
            left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
            left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
            left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
            right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
            right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
            right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])
            left_img = torchvision.transforms.functional.adjust_saturation(left_img, random_saturation[0])
            right_img = torchvision.transforms.functional.adjust_saturation(right_img, random_saturation[1])
            right_img = np.array(right_img)
            left_img = np.array(left_img)

            # geometric unsymmetric-augmentation
            angle = 0;
            px = 0
            if np.random.binomial(1, 0.5):
                # angle = 0.1;
                # px = 2
                angle = 0.05
                px = 1
            co_transform = flow_transforms.Compose([
                # flow_transforms.RandomVdisp(angle, px),
                # flow_transforms.Scale(np.random.uniform(self.rand_scale[0], self.rand_scale[1]), order=self.order),
                flow_transforms.RandomCrop((th, tw)),
            ])
            augmented, disparity = co_transform([left_img, right_img], disparity)
            left_img = augmented[0]
            right_img = augmented[1]

            points = self.RGB2EdgePoints(left_img)
            if points is None:
                points_w = torch.randint(0, tw, (self.num_points, 1), dtype=torch.float32)
                points_h = torch.randint(0, th, (self.num_points, 1), dtype=torch.float32)
                points = torch.cat((points_w, points_h), dim=-1)
            elif points.shape[0] < self.num_points:
                points = torch.from_numpy(points).to(dtype=torch.float32)
                n = points.shape[0]
                points_w = torch.randint(0, tw, (self.num_points - n, 1), dtype=torch.float32)
                points_h = torch.randint(0, th, (self.num_points - n, 1), dtype=torch.float32)
                points_new = torch.cat((points_w, points_h), dim=-1)
                points = torch.cat((points, points_new), dim=0)
            else:
                points_idx = np.arange(0, points.shape[0], dtype=np.int32)
                np.random.shuffle(points_idx)
                points_idx = points_idx[:self.num_points]
                points = points[points_idx, :]
                points = torch.from_numpy(points).to(dtype=torch.float32)

            # right_img.flags.writeable = True
            # if np.random.binomial(1,0.2):
            #   sx = int(np.random.uniform(35,100))
            #   sy = int(np.random.uniform(25,75))
            #   cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
            #   cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
            #   right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]

            # to tensor, normalize
            disparity = np.ascontiguousarray(disparity, dtype=np.float32)

            disparity_low = cv2.resize(disparity, (tw//4, th//4), interpolation=cv2.INTER_NEAREST)
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "disparity_low": disparity_low,
                    "points": points}
        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1248x384
            top_pad = 384 - h
            right_pad = 1248 - w

            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]
                        }
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

from PIL import ImageFilter

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

_GLOBAL_SEED = 0
logger = getLogger()


def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    logger.info('making imagenet data transforms')

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    transform_list = []
    transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class CSINormalize(object):
    """
    Normalize per-channel using given mean/std, or fallback to per-sample standardization.
    """
    def __init__(self, mean=None, std=None, eps=1e-6):
        self.mean = None if mean is None else torch.tensor(mean).view(-1, 1, 1)
        self.std = None if std is None else torch.tensor(std).view(-1, 1, 1)
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C,H,W]
        if self.mean is not None and self.std is not None:
            mean = self.mean.to(dtype=x.dtype, device=x.device)
            std = self.std.to(dtype=x.dtype, device=x.device)
            return (x - mean) / (std + self.eps)

        # per-sample standardization (channel-wise)
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        return (x - mean) / (std + self.eps)


class AddGaussianNoise(object):
    def __init__(self, sigma=0.0):
        self.sigma = float(sigma)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigma <= 0:
            return x
        return x + torch.randn_like(x) * self.sigma


class RandomCropOrPad(object):
    """
    Ensure output is exactly (out_h, out_w).
    If input is larger: random crop.
    If smaller: zero pad.
    """
    def __init__(self, out_h: int, out_w: int):
        self.out_h = int(out_h)
        self.out_w = int(out_w)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape
        # pad if needed
        pad_h = max(0, self.out_h - h)
        pad_w = max(0, self.out_w - w)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # pad right & bottom
            c, h, w = x.shape

        # random crop if needed
        if h > self.out_h:
            top = torch.randint(0, h - self.out_h + 1, (1,)).item()
        else:
            top = 0
        if w > self.out_w:
            left = torch.randint(0, w - self.out_w + 1, (1,)).item()
        else:
            left = 0
        return x[:, top:top + self.out_h, left:left + self.out_w]


def make_csi_transforms(
    crop_size=128,
    noise_sigma=0.0,
    normalization=None,  # dict like {"mean":[...16], "std":[...16]} or None
):
    """
    x is already a tensor [C,H,W], we keep it tensor-only.
    """
    logger.info("making CSI tensor transforms")

    mean = None
    std = None
    if isinstance(normalization, dict):
        mean = normalization.get("mean", None)
        std = normalization.get("std", None)

    def _compose(x):
        x = RandomCropOrPad(crop_size, crop_size)(x)
        x = CSINormalize(mean=mean, std=std)(x)
        x = AddGaussianNoise(sigma=noise_sigma)(x)
        return x

    return _compose

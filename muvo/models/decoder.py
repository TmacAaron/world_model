from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from muvo.models.common import RGBHead, SegmentationHead, DepthHead, SemHead, LidarReHead, LidarSegHead, VoxelSemHead


class PolicyDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(True),
            nn.Linear(in_channels // 2, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.fc(x)


class ConvDecoder2D(nn.Module):
    def __init__(self, input_n_channels, out_n_channels, latent_n_channels=512, n_basic_conv=2,
                 activation=nn.ELU, head='rgb'):
        super().__init__()
        n_channels = latent_n_channels

        # 288 * 10 * 26
        self.pre_transpose_conv = nn.Sequential(
            nn.Conv2d(input_n_channels, n_channels, kernel_size=3, padding=1),
            activation(),
            # nn.Conv2d(n_channels, n_channels, kernel_size=5, padding=2),
            # activation(),
            *[nn.Sequential(
                nn.ConvTranspose2d(n_channels, n_channels, kernel_size=4, stride=2, padding=1),
                activation()) for _ in range(n_basic_conv)],
        )

        head_modules = {'rgb': RGBHead,
                        'bev': SegmentationHead,
                        'depth': DepthHead,
                        'sem_image': SemHead,
                        'lidar_re': LidarReHead,
                        'lidar_seg': LidarSegHead}
        head_module = head_modules[head] if head in head_modules else RGBHead

        self.trans_conv1 = nn.Sequential(
            nn.ConvTranspose2d(n_channels, 256, kernel_size=6, stride=2, padding=2),
            activation(),
        )
        self.head_4 = head_module(in_channels=256, n_classes=out_n_channels, downsample_factor=4)
        # 256 x 80 x 208

        self.trans_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=6, stride=2, padding=2),
            activation(),
        )
        self.head_2 = head_module(in_channels=128, n_classes=out_n_channels, downsample_factor=2)
        # 128 x 160 x 416

        self.trans_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2, padding=2),
            activation()
        )
        self.head_1 = head_module(in_channels=64, n_classes=out_n_channels, downsample_factor=1)
        # 64 x 320 x 832

    def forward(self, x):
        x = self.pre_transpose_conv(x)

        x = self.trans_conv1(x)
        output_4 = self.head_4(x)
        x = self.trans_conv2(x)
        output_2 = self.head_2(x)
        x = self.trans_conv3(x)
        output_1 = self.head_1(x)

        output = {**output_4, **output_2, **output_1}
        return output


class ConvDecoder3D(nn.Module):
    def __init__(self, input_n_channels, out_n_channels, latent_n_channels=64, activation=nn.ELU):
        super().__init__()
        n_channels = latent_n_channels

        # 288 x 12 x 12 x 4
        self.pre_transpose_conv = nn.Sequential(
            nn.Conv3d(input_n_channels, 2 * n_channels, kernel_size=3, padding=1),  # 128 x 12 x 12 x 4
            activation(),
            nn.ConvTranspose3d(2 * n_channels, n_channels, kernel_size=4, stride=2, padding=1),  # 64 x 24 x 24 x 8
            activation(),
        )

        head_module = VoxelSemHead

        self.trans_conv1 = nn.Sequential(
            nn.ConvTranspose3d(n_channels, n_channels // 2, kernel_size=6, stride=2, padding=2),
            activation(),
        )
        self.head_4 = head_module(in_channels=n_channels // 2, n_classes=out_n_channels, downsample_factor=4)
        # 32 x 48 x 48 x 16

        self.trans_conv2 = nn.Sequential(
            nn.ConvTranspose3d(n_channels // 2, n_channels // 4, kernel_size=6, stride=2, padding=2),
            activation(),
        )
        self.head_2 = head_module(in_channels=n_channels // 4, n_classes=out_n_channels, downsample_factor=2)
        # 16 x 96 x 96 x 32

        self.trans_conv3 = nn.Sequential(
            nn.ConvTranspose3d(n_channels // 4, n_channels // 8, kernel_size=6, stride=2, padding=2),
            activation()
        )
        self.head_1 = head_module(in_channels=n_channels // 8, n_classes=out_n_channels, downsample_factor=1)
        # 8 x 192 x 192 x 64

    def forward(self, x):
        x = self.pre_transpose_conv(x)

        x = self.trans_conv1(x)
        output_4 = self.head_4(x)
        x = self.trans_conv2(x)
        output_2 = self.head_2(x)
        x = self.trans_conv3(x)
        output_1 = self.head_1(x)

        output = {**output_4, **output_2, **output_1}
        return output

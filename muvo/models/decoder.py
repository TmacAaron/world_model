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
            # nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            # activation(),
        )
        self.head_4 = head_module(in_channels=256, n_classes=out_n_channels, downsample_factor=4)
        # 256 x 80 x 208

        self.trans_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=6, stride=2, padding=2),
            activation(),
            # nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            # activation(),
        )
        self.head_2 = head_module(in_channels=128, n_classes=out_n_channels, downsample_factor=2)
        # 128 x 160 x 416

        self.trans_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2, padding=2),
            activation(),
            # nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            # activation(),
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
            # nn.Conv3d(n_channels // 2, n_channels // 2, kernel_size=5, stride=1, padding=2),
            # activation(),
        )
        self.head_4 = head_module(in_channels=n_channels // 2, n_classes=out_n_channels, downsample_factor=4)
        # 32 x 48 x 48 x 16

        self.trans_conv2 = nn.Sequential(
            nn.ConvTranspose3d(n_channels // 2, n_channels // 4, kernel_size=6, stride=2, padding=2),
            activation(),
            # nn.Conv3d(n_channels // 4, n_channels // 4, kernel_size=5, stride=1, padding=2),
            # activation(),
        )
        self.head_2 = head_module(in_channels=n_channels // 4, n_classes=out_n_channels, downsample_factor=2)
        # 16 x 96 x 96 x 32

        self.trans_conv3 = nn.Sequential(
            nn.ConvTranspose3d(n_channels // 4, n_channels // 8, kernel_size=6, stride=2, padding=2),
            activation(),
            # nn.Conv3d(n_channels // 8, n_channels // 8, kernel_size=5, stride=1, padding=2),
            # activation(),
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


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_n_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv1 = ConvInstanceNorm(in_channels, out_channels, latent_n_channels)
        self.conv2 = ConvInstanceNorm(out_channels, out_channels, latent_n_channels)

    def forward(self, x, w):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.conv1(x, w)
        return self.conv2(x, w)


class DecoderBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, latent_n_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv1 = ConvInstanceNorm3d(in_channels, out_channels, latent_n_channels)
        self.conv2 = ConvInstanceNorm3d(out_channels, out_channels, latent_n_channels)

    def forward(self, x, w):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='trilinear', align_corners=False)
        x = self.conv1(x, w)
        return self.conv2(x, w)


class ConvInstanceNorm(nn.Module):
    def __init__(self, in_channels, out_channels, latent_n_channels):
        super().__init__()
        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.adaptive_norm = AdaptiveInstanceNorm(latent_n_channels, out_channels)

    def forward(self, x, w):
        x = self.conv_act(x)
        return self.adaptive_norm(x, w)


class ConvInstanceNorm3d(nn.Module):
    def __init__(self, in_channels, out_channels, latent_n_channels):
        super().__init__()
        self.conv_act = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.adaptive_norm = AdaptiveInstanceNorm3d(latent_n_channels, out_channels)

    def forward(self, x, w):
        x = self.conv_act(x)
        return self.adaptive_norm(x, w)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, latent_n_channels, out_channels, epsilon=1e-8):
        super().__init__()
        self.out_channels = out_channels
        self.epsilon = epsilon

        # self.latent_affine = nn.Linear(latent_n_channels, 2 * out_channels)
        self.latent_affine = nn.Conv2d(latent_n_channels, 2 * out_channels, 3, 1, 1)

    def forward(self, x, style):
        #  Instance norm
        mean = x.mean(dim=(-1, -2), keepdim=True)
        x = x - mean
        std = torch.sqrt(torch.mean(x ** 2, dim=(-1, -2), keepdim=True) + self.epsilon)
        x = x / std

        # Normalising with the style vector
        style = F.interpolate(style, size=x.shape[-2:], mode='nearest')
        style = self.latent_affine(style)
        style += style.mean(dim=(-1, -2), keepdim=True)
        scale, bias = torch.split(style, split_size_or_sections=self.out_channels, dim=1)
        out = scale * x + bias
        return out


class AdaptiveInstanceNorm3d(nn.Module):
    def __init__(self, latent_n_channels, out_channels, epsilon=1e-8):
        super().__init__()
        self.out_channels = out_channels
        self.epsilon = epsilon

        # self.latent_affine = nn.Linear(latent_n_channels, 2 * out_channels)
        self.latent_affine = nn.Conv3d(latent_n_channels, 2 * out_channels, 3, 1, 1)

    def forward(self, x, style):
        #  Instance norm
        mean = x.mean(dim=(-1, -2, -3), keepdim=True)
        x = x - mean
        std = torch.sqrt(torch.mean(x ** 2, dim=(-1, -2, -3), keepdim=True) + self.epsilon)
        x = x / std

        # Normalising with the style vector
        style = F.interpolate(style, size=x.shape[-3:], mode='nearest')
        style = self.latent_affine(style)
        style += style.mean(dim=(-1, -2, -3), keepdim=True)
        scale, bias = torch.split(style, split_size_or_sections=self.out_channels, dim=1)
        out = scale * x + bias
        return out


class StyleDecoder2D(nn.Module):
    def __init__(self, latent_n_channels, out_n_channels, constant_size=(10, 26), n_basic_conv=2, head='rgb'):
        super().__init__()
        n_channels = 512

        self.constant_tensor = nn.Parameter(torch.randn((n_channels, *constant_size), dtype=torch.float32))

        # Input 512 x 10 x 26
        self.first_norm = AdaptiveInstanceNorm(latent_n_channels, out_channels=n_channels)
        self.first_conv = ConvInstanceNorm(n_channels, n_channels, latent_n_channels)
        # 512 x 10 x 26

        self.middle_conv = nn.ModuleList(
            [DecoderBlock(n_channels, n_channels, latent_n_channels, upsample=True) for _ in range(n_basic_conv)]
        )

        head_modules = {'rgb': RGBHead,
                        'bev': SegmentationHead,
                        'depth': DepthHead,
                        'sem_image': SemHead,
                        'lidar_re': LidarReHead,
                        'lidar_seg': LidarSegHead}
        head_module = head_modules[head] if head in head_modules else RGBHead
        # 512 x 40 x 104
        self.conv1 = DecoderBlock(n_channels, 256, latent_n_channels, upsample=True)
        self.head_4 = head_module(256, out_n_channels, downsample_factor=4)
        # 256 x 80 x 208

        self.conv2 = DecoderBlock(256, 128, latent_n_channels, upsample=True)
        self.head_2 = head_module(128, out_n_channels, downsample_factor=2)
        # 128 x 160 x 416

        self.conv3 = DecoderBlock(128, 64, latent_n_channels, upsample=True)
        self.head_1 = head_module(64, out_n_channels, downsample_factor=1)
        # 64 x 320 x 832

    def forward(self, w: Tensor) -> Tensor:
        b = w.shape[0]
        x = self.constant_tensor.unsqueeze(0).repeat([b, 1, 1, 1])

        x = self.first_norm(x, w)
        x = self.first_conv(x, w)

        for module in self.middle_conv:
            x = module(x, w)

        x = self.conv1(x, w)
        output_4 = self.head_4(x)
        x = self.conv2(x, w)
        output_2 = self.head_2(x)
        x = self.conv3(x, w)
        output_1 = self.head_1(x)

        output = {**output_4, **output_2, **output_1}
        return output


class StyleDecoder3D(nn.Module):
    def __init__(self, latent_n_channels, out_n_channels, feature_channels=64, constant_size=(12, 12, 4)):
        super().__init__()
        n_channels = feature_channels

        self.constant_tensor = nn.Parameter(torch.randn((2 * n_channels, *constant_size), dtype=torch.float32))

        # Input 12 x 12 x 4
        self.first_norm = AdaptiveInstanceNorm3d(latent_n_channels, out_channels=2 * n_channels)
        self.first_conv = ConvInstanceNorm3d(2 * n_channels, 2 * n_channels, latent_n_channels)
        # 12 x 12 x 4

        self.middle_conv = nn.ModuleList(
            [DecoderBlock3d(2 * n_channels, 2 * n_channels, latent_n_channels, upsample=False),
             DecoderBlock3d(2 * n_channels, n_channels, latent_n_channels, upsample=True)]
        )

        head_module = VoxelSemHead
        # 24 x 24 x 8
        self.conv1 = DecoderBlock3d(n_channels, n_channels // 2, latent_n_channels, upsample=True)
        self.head_4 = head_module(n_channels // 2, out_n_channels, downsample_factor=4)
        # 256 x 48 x 48 x 16

        self.conv2 = DecoderBlock3d(n_channels // 2, n_channels // 4, latent_n_channels, upsample=True)
        self.head_2 = head_module(n_channels // 4, out_n_channels, downsample_factor=2)
        # 128 x 96 x 96 x 32

        self.conv3 = DecoderBlock3d(n_channels // 4, n_channels // 8, latent_n_channels, upsample=True)
        self.head_1 = head_module(n_channels // 8, out_n_channels, downsample_factor=1)
        # 192 x 192 x 64

    def forward(self, w: Tensor) -> Tensor:
        b = w.shape[0]
        x = self.constant_tensor.unsqueeze(0).repeat([b, 1, 1, 1, 1])

        x = self.first_norm(x, w)
        x = self.first_conv(x, w)

        for module in self.middle_conv:
            x = module(x, w)

        x = self.conv1(x, w)
        output_4 = self.head_4(x)
        x = self.conv2(x, w)
        output_2 = self.head_2(x)
        x = self.conv3(x, w)
        output_1 = self.head_1(x)

        output = {**output_4, **output_2, **output_1}
        return output

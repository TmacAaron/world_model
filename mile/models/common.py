from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RouteEncode(nn.Module):
    def __init__(self, out_channels, backbone='resnet18'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True, out_indices=[4])
        self.out_channels = out_channels
        feature_info = self.backbone.feature_info.get_dicts(keys=['num_chs', 'reduction'])
        self.fc = nn.Linear(feature_info[-1]['num_chs'], out_channels)

    def forward(self, route):
        x = self.backbone(route)[0]
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.fc(x)


class GRUCellLayerNorm(nn.Module):
    def __init__(self, input_size, hidden_size, reset_bias=1.0):
        super().__init__()
        self.reset_bias = reset_bias

        self.update_layer = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.update_norm = nn.LayerNorm(hidden_size)

        self.reset_layer = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.reset_norm = nn.LayerNorm(hidden_size)

        self.proposal_layer = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.proposal_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs, state):
        update = self.update_layer(torch.cat([inputs, state], -1))
        update = torch.sigmoid(self.update_norm(update))

        reset = self.reset_layer(torch.cat([inputs, state], -1))
        reset = torch.sigmoid(self.reset_norm(reset) + self.reset_bias)

        h_n = self.proposal_layer(torch.cat([inputs, reset * state], -1))
        h_n = torch.tanh(self.proposal_norm(h_n))
        output = (1 - update) * h_n + update * state
        return output


class Policy(nn.Module):
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


class Decoder(nn.Module):
    def __init__(self, feature_info, out_channels):
        super().__init__()
        n_upsample_skip_convs = len(feature_info) - 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(feature_info[-1]['num_chs'], out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.upsample_skip_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(feature_info[-i]['num_chs'], out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
            for i in range(2, n_upsample_skip_convs + 2)
        )

        self.out_channels = out_channels

    def forward(self, xs: List[Tensor]) -> Tensor:
        x = self.conv1(xs[-1])

        for i, conv in enumerate(self.upsample_skip_convs):
            size = xs[-(i + 2)].shape[-2:]
            x = conv(xs[-(i + 2)]) + F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        return x


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


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, latent_n_channels, out_channels, epsilon=1e-8):
        super().__init__()
        self.out_channels = out_channels
        self.epsilon = epsilon

        self.latent_affine = nn.Linear(latent_n_channels, 2 * out_channels)

    def forward(self, x, style):
        #  Instance norm
        mean = x.mean(dim=(-1, -2), keepdim=True)
        x = x - mean
        std = torch.sqrt(torch.mean(x ** 2, dim=(-1, -2), keepdim=True) + self.epsilon)
        x = x / std

        # Normalising with the style vector
        style = self.latent_affine(style).unsqueeze(-1).unsqueeze(-1)
        scale, bias = torch.split(style, split_size_or_sections=self.out_channels, dim=1)
        out = scale * x + bias
        return out


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, n_classes, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = {
            f'bev_segmentation_{self.downsample_factor}': self.segmentation_head(x),
            f'bev_instance_offset_{self.downsample_factor}': self.instance_offset_head(x),
            f'bev_instance_center_{self.downsample_factor}': self.instance_center_head(x),
        }
        return output


class RGBHead(nn.Module):
    def __init__(self, in_channels, n_classes, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.rgb_head = nn.Sequential(
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0),
        )

    def forward(self, x):
        output = {
            f'rgb_{self.downsample_factor}': self.rgb_head(x),
        }
        return output


class BevDecoder(nn.Module):
    def __init__(self, latent_n_channels, semantic_n_channels, constant_size=(3, 3), is_segmentation=True):
        super().__init__()
        n_channels = 512

        self.constant_tensor = nn.Parameter(torch.randn((n_channels, *constant_size), dtype=torch.float32))

        # Input 512 x 3 x 3
        self.first_norm = AdaptiveInstanceNorm(latent_n_channels, out_channels=n_channels)
        self.first_conv = ConvInstanceNorm(n_channels, n_channels, latent_n_channels)
        # 512 x 3 x 3

        self.middle_conv = nn.ModuleList(
            [DecoderBlock(n_channels, n_channels, latent_n_channels, upsample=True) for _ in range(3)]
        )

        head_module = SegmentationHead if is_segmentation else RGBHead
        # 512 x 24 x 24
        self.conv1 = DecoderBlock(n_channels, 256, latent_n_channels, upsample=True)
        self.head_4 = head_module(256, semantic_n_channels, downsample_factor=4)
        # 256 x 48 x 48

        self.conv2 = DecoderBlock(256, 128, latent_n_channels, upsample=True)
        self.head_2 = head_module(128, semantic_n_channels, downsample_factor=2)
        # 128 x 96 x 96

        self.conv3 = DecoderBlock(128, 64, latent_n_channels, upsample=True)
        self.head_1 = head_module(64, semantic_n_channels, downsample_factor=1)
        # 64 x 192 x 192

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


class VoxelDecoderScale(nn.Module):
    def __init__(self, input_channels, n_classes, kernel_size=1, feature_channels=512):
        super().__init__()

        self.weight_xy_decoder = nn.Conv2d(input_channels, 1, kernel_size, 1)
        self.weight_xz_decoder = nn.Conv2d(input_channels, 1, kernel_size, 1)
        self.weight_yz_decoder = nn.Conv2d(input_channels, 1, kernel_size, 1)

        self.classifier = nn.Sequential(
            nn.Linear(feature_channels, feature_channels),
            nn.Softplus(),
            nn.Linear(feature_channels, n_classes)
        )

    def attention_fusion(self, t1, w1, t2, w2):
        norm_weight = torch.softmax(torch.cat([w1, w2], dim=1), dim=1)
        feat = t1 * norm_weight[:, 0:1] + t2 * norm_weight[:, 1:2]
        return feat

    def expand_to_XYZ(self, xy_feat, xz_feat, yz_feat):
        B, C, X, Y, Z = *xy_feat.size(), xz_feat.size(3)
        xy_feat = xy_feat.view(B, C, X, Y, 1)
        xz_feat = xz_feat.view(B, C, X, 1, Z)
        yz_feat = yz_feat.view(B, C, 1, Y, Z)
        return torch.broadcast_tensors(xy_feat, xz_feat, yz_feat)

    def forward(self, x):
        feature_xy, feature_xz, feature_yz = x

        weights_xy = self.weight_xy_decoder(feature_xy)
        weights_xz = self.weight_xz_decoder(feature_xz)
        weights_yz = self.weight_yz_decoder(feature_yz)

        feature_xy, feature_xz, feature_yz = self.expand_to_XYZ(feature_xy, feature_xz, feature_yz)
        weights_xy, weights_xz, weights_yz = self.expand_to_XYZ(weights_xy, weights_xz, weights_yz)

        features_xyz = self.attention_fusion(feature_xy, weights_xy, feature_xz, weights_xz) + \
                       self.attention_fusion(feature_xy, weights_xy, feature_yz, weights_yz)

        B, C, X, Y, Z = features_xyz.size()
        logits = self.classifier(features_xyz.view(B, C, -1).transpose(1, 2))
        logits = logits.permute(0, 2, 1).reshape(B, -1, X, Y, Z)

        return logits


class VoxelDecoder(nn.Module):
    def __init__(self, input_channels, n_classes, kernel_size=1, feature_channels=512):
        super().__init__()

        self.decoder_1 = VoxelDecoderScale(input_channels, n_classes, kernel_size, feature_channels)
        self.decoder_2 = VoxelDecoderScale(input_channels, n_classes, kernel_size, feature_channels)
        self.decoder_4 = VoxelDecoderScale(input_channels, n_classes, kernel_size, feature_channels)

    def forward(self, xy, xz, yz):
        output_1 = self.decoder_1((xy['rgb_1'], xz['rgb_1'], yz['rgb_1']))
        output_2 = self.decoder_2((xy['rgb_2'], xz['rgb_2'], yz['rgb_2']))
        output_4 = self.decoder_4((xy['rgb_4'], xz['rgb_4'], yz['rgb_4']))
        return {'voxel_1': output_1,
                'voxel_2': output_2,
                'voxel_4': output_4}


class LidarDecoder(nn.Module):
    def __init__(self, latent_n_channels, semantic_n_channels, constant_size=(3, 3), is_segmentation=True):
        super().__init__()
        self.is_seg = is_segmentation
        self.decoder = BevDecoder(latent_n_channels, semantic_n_channels, constant_size, is_segmentation=False)

    def forward(self, x):
        output = self.decoder(x)
        if self.is_seg:
            return{
                'lidar_segmentation_1': output['rgb_1'],
                'lidar_segmentation_2': output['rgb_2'],
                'lidar_segmentation_4': output['rgb_4'],
            }
        else:
            return {
                'lidar_reconstruction_1': output['rgb_1'],
                'lidar_reconstruction_2': output['rgb_2'],
                'lidar_reconstruction_4': output['rgb_4']
            }

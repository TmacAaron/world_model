import os

import numpy as np
import torch
import lightning.pytorch as pl
from torchmetrics import JaccardIndex

from mile.config import get_cfg
from mile.models.mile import Mile
from mile.losses import SegmentationLoss, KLLoss, RegressionLoss, SpatialRegressionLoss, VoxelLoss, SSIMLoss
from mile.models.preprocess import PreProcess
from constants import BIRDVIEW_COLOURS, VOXEL_COLOURS

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class WorldModelTrainer(pl.LightningModule):
    def __init__(self, hparams, path_to_conf_file=None, pretrained_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = get_cfg(cfg_dict=hparams)
        if path_to_conf_file:
            self.cfg.merge_from_file(path_to_conf_file)
        if pretrained_path:
            self.cfg.PRETRAINED.PATH = pretrained_path
        # print(self.cfg)
        self.vis_step = -1

        self.preprocess = PreProcess(self.cfg)

        # Model
        self.model = Mile(self.cfg)
        self.load_pretrained_weights()

        # Losses
        self.action_loss = RegressionLoss(norm=1)
        if self.cfg.MODEL.TRANSITION.ENABLED:
            self.probabilistic_loss = KLLoss(alpha=self.cfg.LOSSES.KL_BALANCING_ALPHA)

        if self.cfg.SEMANTIC_SEG.ENABLED:
            self.segmentation_loss = SegmentationLoss(
                use_top_k=self.cfg.SEMANTIC_SEG.USE_TOP_K,
                top_k_ratio=self.cfg.SEMANTIC_SEG.TOP_K_RATIO,
                use_weights=self.cfg.SEMANTIC_SEG.USE_WEIGHTS,
                is_bev=True,
                )

            self.center_loss = SpatialRegressionLoss(norm=2)
            self.offset_loss = SpatialRegressionLoss(norm=1, ignore_index=self.cfg.INSTANCE_SEG.IGNORE_INDEX)

            self.metric_iou_val = JaccardIndex(
                task='multiclass', num_classes=self.cfg.SEMANTIC_SEG.N_CHANNELS, average='none',
            )

        if self.cfg.EVAL.RGB_SUPERVISION:
            self.rgb_loss = SpatialRegressionLoss(norm=1)
            if self.cfg.LOSSES.RGB_INSTANCE:
                self.rgb_instance_loss = SpatialRegressionLoss(norm=1)
            if self.cfg.LOSSES.SSIM:
                self.ssim_loss = SSIMLoss(channel=3)

        if self.cfg.LIDAR_RE.ENABLED:
            self.lidar_re_loss = SpatialRegressionLoss(norm=2)

        if self.cfg.LIDAR_SEG.ENABLED:
            self.lidar_seg_loss = SegmentationLoss(
                use_top_k=self.cfg.LIDAR_SEG.USE_TOP_K,
                top_k_ratio=self.cfg.LIDAR_SEG.TOP_K_RATIO,
                use_weights=self.cfg.LIDAR_SEG.USE_WEIGHTS,
                is_bev=False,
            )

        if self.cfg.VOXEL_SEG.ENABLED:
            self.voxel_loss = VoxelLoss(
                use_top_k=self.cfg.VOXEL_SEG.USE_TOP_K,
                top_k_ratio=self.cfg.VOXEL_SEG.TOP_K_RATIO,
                use_weights=self.cfg.VOXEL_SEG.USE_WEIGHTS,
            )

    def load_pretrained_weights(self):
        if self.cfg.PRETRAINED.PATH:
            if os.path.isfile(self.cfg.PRETRAINED.PATH):
                checkpoint = torch.load(self.cfg.PRETRAINED.PATH, map_location='cpu')['state_dict']
                checkpoint = {key[6:]: value for key, value in checkpoint.items() if key[:5] == 'model'}

                self.model.load_state_dict(checkpoint, strict=True)
                print(f'Loaded weights from: {self.cfg.PRETRAINED.PATH}')
            else:
                raise FileExistsError(self.cfg.PRETRAINED.PATH)

    def forward(self, batch, deployment=False):
        batch = self.preprocess(batch)
        output = self.model.forward(batch, deployment=deployment)
        return output

    def deployment_forward(self, batch, is_dreaming):
        batch = self.preprocess(batch)
        output = self.model.deployment_forward(batch, is_dreaming)
        return output

    def shared_step(self, batch):
        output = self.forward(batch)

        losses = dict()

        action_weight = self.cfg.LOSSES.WEIGHT_ACTION
        losses['throttle_brake'] = action_weight * self.action_loss(output['throttle_brake'],
                                                                    batch['throttle_brake'])
        losses['steering'] = action_weight * self.action_loss(output['steering'], batch['steering'])

        if self.cfg.MODEL.TRANSITION.ENABLED:
            probabilistic_loss = self.probabilistic_loss(output['prior'], output['posterior'])

            losses['probabilistic'] = self.cfg.LOSSES.WEIGHT_PROBABILISTIC * probabilistic_loss

        if self.cfg.SEMANTIC_SEG.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                bev_segmentation_loss = self.segmentation_loss(
                    prediction=output[f'bev_segmentation_{downsampling_factor}'],
                    target=batch[f'birdview_label_{downsampling_factor}'],
                )
                discount = 1/downsampling_factor
                losses[f'bev_segmentation_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_SEGMENTATION * \
                                                                    bev_segmentation_loss

                center_loss = self.center_loss(
                    prediction=output[f'bev_instance_center_{downsampling_factor}'],
                    target=batch[f'center_label_{downsampling_factor}']
                )
                offset_loss = self.offset_loss(
                    prediction=output[f'bev_instance_offset_{downsampling_factor}'],
                    target=batch[f'offset_label_{downsampling_factor}']
                )

                center_loss = self.cfg.INSTANCE_SEG.CENTER_LOSS_WEIGHT * center_loss
                offset_loss = self.cfg.INSTANCE_SEG.OFFSET_LOSS_WEIGHT * offset_loss

                losses[f'bev_center_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_INSTANCE * center_loss
                # Offset are already discounted in the labels
                losses[f'bev_offset_{downsampling_factor}'] = self.cfg.LOSSES.WEIGHT_INSTANCE * offset_loss

        if self.cfg.EVAL.RGB_SUPERVISION:
            for downsampling_factor in [1, 2, 4]:
                rgb_weight = 0.1
                discount = 1 / downsampling_factor
                rgb_loss = self.rgb_loss(
                    prediction=output[f'rgb_{downsampling_factor}'],
                    target=batch[f'rgb_label_{downsampling_factor}'],
                )
                if self.cfg.LOSSES.RGB_INSTANCE:
                    rgb_instance_loss = self.rgb_instance_loss(
                        prediction=output[f'rgb_{downsampling_factor}'],
                        target=batch[f'rgb_label_{downsampling_factor}'],
                        instance_mask=batch[f'image_instance_mask_{downsampling_factor}']
                    )
                else:
                    rgb_instance_loss = 0

                if self.cfg.LOSSES.SSIM:
                    ssim_loss = self.ssim_loss(
                        prediction=output[f'rgb_{downsampling_factor}'],
                        target=batch[f'rgb_label_{downsampling_factor}'],
                    )
                    ssim_weight = 0.6
                    losses[f'ssim_{downsampling_factor}'] = rgb_weight * discount * ssim_loss * ssim_weight
                else:
                    ssim_weight = 0

                losses[f'rgb_{downsampling_factor}'] = \
                    rgb_weight * discount * (rgb_loss + 0.5 * rgb_instance_loss) * (1 - ssim_weight)

        if self.cfg.LIDAR_RE.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                discount = 1 / downsampling_factor
                lidar_re_loss = self.lidar_re_loss(
                    prediction=output[f'lidar_reconstruction_{downsampling_factor}'],
                    target=batch[f'range_view_label_{downsampling_factor}']
                )
                losses[f'lidar_re_{downsampling_factor}'] = lidar_re_loss * discount * self.cfg.LOSSES.WEIGHT_LIDAR_RE

        if self.cfg.LIDAR_SEG.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                discount = 1 / downsampling_factor
                lidar_seg_loss = self.lidar_seg_loss(
                    prediction=output[f'lidar_segmentation_{downsampling_factor}'],
                    target=batch[f'range_view_seg_label_{downsampling_factor}']
                )
                losses[f'lidar_seg_{downsampling_factor}'] = \
                    lidar_seg_loss * discount * self.cfg.LOSSES.WEIGHT_LIDAR_SEG

        if self.cfg.VOXEL_SEG.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                discount = 1 / downsampling_factor
                voxel_loss = self.voxel_loss(
                    prediction=output[f'voxel_{downsampling_factor}'],
                    target=batch[f'voxel_label_{downsampling_factor}'].type(torch.long)
                )
                losses[f'voxel_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_VOXEL * voxel_loss

        if self.cfg.MODEL.REWARD.ENABLED:
            reward_loss = self.action_loss(output['reward'], batch['reward'])
            losses['reward'] = self.cfg.LOSSES.WEIGHT_REWARD * reward_loss

        return losses, output

    def training_step(self, batch, batch_idx):
        if batch_idx == self.cfg.STEPS // 2 and self.cfg.MODEL.TRANSITION.ENABLED:
            print('!'*50)
            print('ACTIVE INFERENCE ACTIVATED')
            print('!'*50)
            self.model.rssm.active_inference = True
        losses, output = self.shared_step(batch)

        self.logging_and_visualisation(batch, output, losses, batch_idx, prefix='train')

        return self.loss_reducing(losses)

    def validation_step(self, batch, batch_idx):
        loss, output = self.shared_step(batch)

        if self.cfg.SEMANTIC_SEG.ENABLED:
            seg_prediction = output['bev_segmentation_1'].detach()
            seg_prediction = torch.argmax(seg_prediction, dim=2)
            self.metric_iou_val(
                seg_prediction.view(-1),
                batch['birdview_label'].view(-1)
            )

        self.logging_and_visualisation(batch, output, loss, batch_idx, prefix='val')

        return {'val_loss': self.loss_reducing(loss)}

    def logging_and_visualisation(self, batch, output, loss, batch_idx, prefix='train'):
        # Logging
        self.log('-global_step', torch.tensor(-self.global_step, dtype=torch.float32))
        for key, value in loss.items():
            self.log(f'{prefix}_{key}', value)

        if 'ssim_1' in loss.keys():
            self.log(f'{prefix}_ssim_value', 1 - loss['ssim_1'] / 0.06)

        # Visualisation
        if prefix == 'train':
            visualisation_criteria = (self.global_step % self.cfg.LOG_VIDEO_INTERVAL == 0) \
                                   & (self.global_step != self.vis_step)
            self.vis_step = self.global_step
        else:
            visualisation_criteria = batch_idx == 0
        if visualisation_criteria:
            self.visualise(batch, output, batch_idx, prefix=prefix)

    def loss_reducing(self, loss):
        total_loss = sum([x for x in loss.values()])
        return total_loss

    def on_validation_epoch_end(self):
        class_names = ['Background', 'Road', 'Lane marking', 'Vehicle', 'Pedestrian', 'Green light', 'Yellow light',
                       'Red light and stop sign']
        if self.cfg.SEMANTIC_SEG.ENABLED:
            scores = self.metric_iou_val.compute()
            for key, value in zip(class_names, scores):
                self.logger.experiment.add_scalar('val_iou_' + key, value, global_step=self.global_step)
            self.logger.experiment.add_scalar('val_mean_iou', torch.mean(scores), global_step=self.global_step)
            self.metric_iou_val.reset()

    def visualise(self, batch, output, batch_idx, prefix='train'):
        if not self.cfg.SEMANTIC_SEG.ENABLED:
            return

        target = batch['birdview_label'][:, :, 0]
        pred = torch.argmax(output['bev_segmentation_1'].detach(), dim=-3)

        colours = torch.tensor(BIRDVIEW_COLOURS, dtype=torch.uint8, device=pred.device)

        target = colours[target]
        pred = colours[pred]

        # Move channel to third position
        target = target.permute(0, 1, 4, 2, 3)
        pred = pred.permute(0, 1, 4, 2, 3)

        visualisation_video = torch.cat([target, pred], dim=-1).detach()

        # Rotate for visualisation
        visualisation_video = torch.rot90(visualisation_video, k=1, dims=[3, 4])

        name = f'{prefix}_outputs'
        if prefix == 'val':
            name = name + f'_{batch_idx}'
        self.logger.experiment.add_video(name, visualisation_video, global_step=self.global_step, fps=2)

        if self.cfg.EVAL.RGB_SUPERVISION:
            rgb_target = batch['rgb_label_1']
            rgb_pred = output['rgb_1'].detach()

            visualisation_rgb = torch.cat([rgb_pred, rgb_target], dim=-2).detach()
            name_ = f'{name}_rgb'
            self.logger.experiment.add_video(name_, visualisation_rgb, global_step=self.global_step, fps=2)

        if self.cfg.LIDAR_RE.ENABLED:
            lidar_target = batch['range_view_label_1'][:, :, -1, :, :]
            lidar_pred = output['lidar_reconstruction_1'].detach()[:, :, -1, :, :]

            visualisation_lidar = torch.cat([lidar_pred, lidar_target], dim=-2).detach().unsqueeze(-3)
            name_ = f'{name}_lidar'
            self.logger.experiment.add_video(name_, visualisation_lidar, global_step=self.global_step, fps=2)

        if self.cfg.LIDAR_SEG.ENABLED:
            lidar_seg_target = batch['range_view_seg_label_1'][:, :, 0]
            lidar_seg_pred = torch.argmax(output['lidar_segmentation_1'].detach(), dim=-3)

            colours = torch.tensor(VOXEL_COLOURS, dtype=torch.uint8, device=lidar_seg_pred.device)
            lidar_seg_target = colours[lidar_seg_target]
            lidar_seg_pred = colours[lidar_seg_pred]

            lidar_seg_target = lidar_seg_target.permute(0, 1, 4, 2, 3)
            lidar_seg_pred = lidar_seg_pred.permute(0, 1, 4, 2, 3)

            visualisation_lidar_seg = torch.cat([lidar_seg_pred, lidar_seg_target], dim=-2).detach()
            name_ = f'{name}_lidar_seg'
            self.logger.experiment.add_video(name_, visualisation_lidar_seg, global_step=self.global_step, fps=2)

        if self.cfg.VOXEL_SEG.ENABLED:
            voxel_target = batch['voxel_label_1'][0, 0, 0].cpu().numpy()
            voxel_pred = torch.argmax(output['voxel_1'].detach(), dim=-4).cpu().numpy()[0, 0]
            colours = np.asarray(VOXEL_COLOURS, dtype=float) / 255.0
            voxel_color_target = colours[voxel_target]
            voxel_color_pred = colours[voxel_pred]
            name_ = f'{name}_voxel'
            self.write_voxel_figure(voxel_target, voxel_color_target, f'{name_}_target')
            self.write_voxel_figure(voxel_pred, voxel_color_pred, f'{name_}_pred')

    def write_voxel_figure(self, voxel, voxel_color, name):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.voxels(voxel, facecolors=voxel_color, shade=False)
        ax.view_init(elev=60, azim=165)
        ax.set_axis_off()
        self.logger.experiment.add_figure(name, fig, global_step=self.global_step)

    def configure_optimizers(self):
        #  Do not decay batch norm parameters and biases
        # https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/2
        def add_weight_decay(model, weight_decay=0.01, skip_list=[]):
            no_decay = []
            decay = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or any(x in name for x in skip_list):
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay},
            ]

        parameters = add_weight_decay(
            self.model,
            self.cfg.OPTIMIZER.WEIGHT_DECAY,
            skip_list=['relative_position_bias_table'],
        )
        weight_decay = 0.
        optimizer = torch.optim.AdamW(parameters, lr=self.cfg.OPTIMIZER.LR, weight_decay=weight_decay)

        # scheduler
        if self.cfg.SCHEDULER.NAME == 'none':
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda lr: 1)
        elif self.cfg.SCHEDULER.NAME == 'OneCycleLR':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.cfg.OPTIMIZER.LR,
                total_steps=self.cfg.STEPS,
                pct_start=self.cfg.SCHEDULER.PCT_START,
            )

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

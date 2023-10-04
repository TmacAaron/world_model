import os

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics import JaccardIndex

from mile.config import get_cfg
from mile.models.mile import Mile
from mile.losses import \
    SegmentationLoss, KLLoss, RegressionLoss, SpatialRegressionLoss, VoxelLoss, SSIMLoss, SemScalLoss, GeoScalLoss, \
    CDLoss
from mile.metrics import SSCMetrics, SSIMMetric, CDMetric
from mile.models.preprocess import PreProcess
from mile.utils.geometry_utils import PointCloud
from constants import BIRDVIEW_COLOURS, VOXEL_COLOURS, VOXEL_LABEL

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
        self.rf = self.cfg.RECEPTIVE_FIELD

        self.cml_logger = None
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
            # self.metric_iou_train = JaccardIndex(
            #     task='multiclass', num_classes=self.cfg.SEMANTIC_SEG.N_CHANNELS, average='none',
            # )
            self.metric_iou_test = JaccardIndex(
                task='multiclass', num_classes=self.cfg.SEMANTIC_SEG.N_CHANNELS, average='none',
            )

        if self.cfg.EVAL.RGB_SUPERVISION:
            self.rgb_loss = SpatialRegressionLoss(norm=1)
            if self.cfg.LOSSES.RGB_INSTANCE:
                self.rgb_instance_loss = SpatialRegressionLoss(norm=1)
            if self.cfg.LOSSES.SSIM:
                self.ssim_loss = SSIMLoss(channel=3)
            self.ssim_metric_val = SSIMMetric(channel=3)
            # self.ssim_metric_train = SSIMMetric(channel=3)
            self.ssim_metric_test = SSIMMetric(channel=3)

        if self.cfg.LIDAR_RE.ENABLED:
            self.lidar_re_loss = SpatialRegressionLoss(norm=2)
            self.lidar_depth_loss = SpatialRegressionLoss(norm=1)
            # self.lidar_cd_loss = CDLoss()
            self.pcd = PointCloud(
                self.cfg.POINTS.CHANNELS,
                self.cfg.POINTS.HORIZON_RESOLUTION,
                *self.cfg.POINTS.FOV,
                self.cfg.POINTS.LIDAR_POSITION
            )
            self.cd_metric_test = CDMetric()
            self.cd_metric_val = CDMetric()

        if self.cfg.LIDAR_SEG.ENABLED:
            self.lidar_seg_loss = SegmentationLoss(
                use_top_k=self.cfg.LIDAR_SEG.USE_TOP_K,
                top_k_ratio=self.cfg.LIDAR_SEG.TOP_K_RATIO,
                use_weights=self.cfg.LIDAR_SEG.USE_WEIGHTS,
                is_bev=False,
            )

        if self.cfg.SEMANTIC_IMAGE.ENABLED:
            self.sem_image_loss = SegmentationLoss(
                use_top_k=self.cfg.SEMANTIC_IMAGE.USE_TOP_K,
                top_k_ratio=self.cfg.SEMANTIC_IMAGE.TOP_K_RATIO,
                use_weights=self.cfg.SEMANTIC_IMAGE.USE_WEIGHTS,
                is_bev=False,
            )

        if self.cfg.DEPTH.ENABLED:
            self.depth_image_loss = SpatialRegressionLoss(norm=1)

        if self.cfg.VOXEL_SEG.ENABLED:
            self.voxel_loss = VoxelLoss(
                use_top_k=self.cfg.VOXEL_SEG.USE_TOP_K,
                top_k_ratio=self.cfg.VOXEL_SEG.TOP_K_RATIO,
                use_weights=self.cfg.VOXEL_SEG.USE_WEIGHTS,
            )
            self.sem_scal_loss = SemScalLoss()
            self.geo_scal_loss = GeoScalLoss()
            # self.ssc_metrics_train = SSCMetrics(self.cfg.VOXEL_SEG.N_CLASSES)
            self.ssc_metrics_val = SSCMetrics(self.cfg.VOXEL_SEG.N_CLASSES)
            self.ssc_metrics_test = SSCMetrics(self.cfg.VOXEL_SEG.N_CLASSES)

    def get_cml_logger(self, cml_logger):
        self.cml_logger = cml_logger

    def load_pretrained_weights(self):
        if self.cfg.PRETRAINED.PATH:
            if os.path.isfile(self.cfg.PRETRAINED.PATH):
                checkpoint = torch.load(self.cfg.PRETRAINED.PATH, map_location='cpu')['state_dict']
                checkpoint = {key[6:]: value for key, value in checkpoint.items() if key[:5] == 'model'}

                self.model.load_state_dict(checkpoint, strict=False)
                print(f'Loaded weights from: {self.cfg.PRETRAINED.PATH}')
            else:
                raise FileExistsError(self.cfg.PRETRAINED.PATH)

    def forward(self, batch, deployment=False, predict_action=False):
        batch = self.preprocess(batch)
        output, output_imagine = self.model.forward(batch, deployment=deployment, predict_action=predict_action)
        return output, output_imagine

    def deployment_forward(self, batch, is_dreaming):
        batch = self.preprocess(batch)
        output = self.model.deployment_forward(batch, is_dreaming)
        return output

    def shared_step(self, batch):
        output, output_imagine = self.forward(batch)
        train_imagine = False
        if train_imagine:
            end_ind = None
            output_train = {key: torch.cat([output[key], output_imagine[key]], dim=1) for key in output.keys()}
        else:
            end_ind = self.rf
            output_train = output

        losses = dict()

        action_weight = self.cfg.LOSSES.WEIGHT_ACTION
        losses['throttle_brake'] = action_weight * self.action_loss(output_train['throttle_brake'],
                                                                    batch['throttle_brake'][:, :end_ind])
        losses['steering'] = action_weight * self.action_loss(output_train['steering'], batch['steering'][:, :end_ind])

        if self.cfg.MODEL.TRANSITION.ENABLED:
            probabilistic_loss = self.probabilistic_loss(output_train['prior'], output_train['posterior'])

            losses['probabilistic'] = self.cfg.LOSSES.WEIGHT_PROBABILISTIC * probabilistic_loss

        if self.cfg.SEMANTIC_SEG.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                bev_segmentation_loss = self.segmentation_loss(
                    prediction=output_train[f'bev_segmentation_{downsampling_factor}'],
                    target=batch[f'birdview_label_{downsampling_factor}'][:, :end_ind],
                )
                discount = 1 / downsampling_factor
                losses[f'bev_segmentation_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_SEGMENTATION * \
                                                                    bev_segmentation_loss

                center_loss = self.center_loss(
                    prediction=output_train[f'bev_instance_center_{downsampling_factor}'],
                    target=batch[f'center_label_{downsampling_factor}'][:, :end_ind]
                )
                offset_loss = self.offset_loss(
                    prediction=output_train[f'bev_instance_offset_{downsampling_factor}'],
                    target=batch[f'offset_label_{downsampling_factor}'][:, :end_ind]
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
                    prediction=output_train[f'rgb_{downsampling_factor}'],
                    target=batch[f'rgb_label_{downsampling_factor}'][:, :end_ind],
                )

                if self.cfg.LOSSES.RGB_INSTANCE:
                    rgb_instance_loss = self.rgb_instance_loss(
                        prediction=output_train[f'rgb_{downsampling_factor}'],
                        target=batch[f'rgb_label_{downsampling_factor}'][:, :end_ind],
                        instance_mask=batch[f'image_instance_mask_{downsampling_factor}'][:, :end_ind]
                    )
                else:
                    rgb_instance_loss = 0

                if self.cfg.LOSSES.SSIM:
                    ssim_loss = 1 - self.ssim_loss(
                        prediction=output_train[f'rgb_{downsampling_factor}'],
                        target=batch[f'rgb_label_{downsampling_factor}'][:, :end_ind],
                    )
                    ssim_weight = 0.6
                    losses[f'ssim_{downsampling_factor}'] = rgb_weight * discount * ssim_loss * ssim_weight

                losses[f'rgb_{downsampling_factor}'] = \
                    rgb_weight * discount * (rgb_loss + 0.5 * rgb_instance_loss)

        if self.cfg.LIDAR_RE.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                discount = 1 / downsampling_factor
                lidar_re_loss = self.lidar_re_loss(
                    prediction=output_train[f'lidar_reconstruction_{downsampling_factor}'][:, :, :3, :, :],
                    target=batch[f'range_view_label_{downsampling_factor}'][:, :end_ind][:, :, :3, :, :]
                )
                lidar_depth_loss = self.lidar_depth_loss(
                    prediction=output_train[f'lidar_reconstruction_{downsampling_factor}'][:, :, -1:, :, :],
                    target=batch[f'range_view_label_{downsampling_factor}'][:, :end_ind][:, :, -1:, :, :]
                )
                losses[f'lidar_re_{downsampling_factor}'] = lidar_re_loss * discount * self.cfg.LOSSES.WEIGHT_LIDAR_RE
                losses[f'lidar_depth_{downsampling_factor}'] = lidar_depth_loss * discount * self.cfg.LOSSES.WEIGHT_LIDAR_RE

        if self.cfg.LIDAR_SEG.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                discount = 1 / downsampling_factor
                lidar_seg_loss = self.lidar_seg_loss(
                    prediction=output_train[f'lidar_segmentation_{downsampling_factor}'],
                    target=batch[f'range_view_seg_label_{downsampling_factor}'][:, :end_ind]
                )
                losses[f'lidar_seg_{downsampling_factor}'] = \
                    lidar_seg_loss * discount * self.cfg.LOSSES.WEIGHT_LIDAR_SEG

        if self.cfg.SEMANTIC_IMAGE.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                discount = 1 / downsampling_factor
                sem_image_loss = self.sem_image_loss(
                    prediction=output_train[f'semantic_image_{downsampling_factor}'],
                    target=batch[f'semantic_image_label_{downsampling_factor}'][:, :end_ind]
                )
                losses[f'semantic_image_{downsampling_factor}'] = \
                    sem_image_loss * discount * self.cfg.LOSSES.WEIGHT_SEM_IMAGE

        if self.cfg.DEPTH.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                discount = 1 / downsampling_factor
                depth_image_loss = self.depth_image_loss(
                    prediction=output_train[f'depth_{downsampling_factor}'],
                    target=batch[f'depth_label_{downsampling_factor}'][:, :end_ind]
                )
                losses[f'depth_{downsampling_factor}'] = \
                    depth_image_loss * discount * self.cfg.LOSSES.WEIGHT_DEPTH

        if self.cfg.VOXEL_SEG.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                discount = 1 / downsampling_factor
                voxel_loss = self.voxel_loss(
                    prediction=output_train[f'voxel_{downsampling_factor}'],
                    target=batch[f'voxel_label_{downsampling_factor}'][:, :end_ind].type(torch.long)
                )
                sem_scal_loss = self.sem_scal_loss(
                    prediction=output_train[f'voxel_{downsampling_factor}'],
                    target=batch[f'voxel_label_{downsampling_factor}'][:, :end_ind]
                )
                geo_scal_loss = self.geo_scal_loss(
                    prediction=output_train[f'voxel_{downsampling_factor}'],
                    target=batch[f'voxel_label_{downsampling_factor}'][:, :end_ind]
                )
                losses[f'voxel_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_VOXEL * voxel_loss
                losses[f'sem_scal_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_VOXEL * sem_scal_loss
                losses[f'geo_scal_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_VOXEL * geo_scal_loss

        if self.cfg.MODEL.REWARD.ENABLED:
            reward_loss = self.action_loss(output_train['reward'], batch['reward'][:, :end_ind])
            losses['reward'] = self.cfg.LOSSES.WEIGHT_REWARD * reward_loss

        return losses, output, output_imagine

    def compute_ssc_metrics(self, batch, output, metric):
        y_true = batch['voxel_label_1'][:, :self.rf].cpu().numpy()
        y_pred = output['voxel_1'].detach().cpu().numpy()
        b, s, c, x, y, z = y_pred.shape
        y_pred = y_pred.reshape(b * s, c, x, y, z)
        y_true = y_true.reshape(b * s, x, y, z)
        y_pred = np.argmax(y_pred, axis=1)
        metric.add_batch(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        if batch_idx == self.cfg.STEPS // 2 and self.cfg.MODEL.TRANSITION.ENABLED:
            print('!' * 50)
            print('ACTIVE INFERENCE ACTIVATED')
            print('!' * 50)
            self.model.rssm.active_inference = True
        losses, output, output_imagine = self.shared_step(batch)

        if self.cfg.VOXEL_SEG.ENABLED:
            self.compute_ssc_metrics(batch, output, self.ssc_metrics_train)
            # self.ssc_metrics_train.get_stats()

        self.logging_and_visualisation(batch, output, output_imagine, losses, batch_idx, prefix='train')

        return self.loss_reducing(losses)

    def validation_step(self, batch, batch_idx):
        self.train()
        for module in self.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        with torch.no_grad():
            loss, output, output_imagine = self.shared_step(batch)
        self.eval()

        if self.cfg.SEMANTIC_SEG.ENABLED:
            seg_prediction = output['bev_segmentation_1'].detach()
            seg_prediction = torch.argmax(seg_prediction, dim=2)
            self.metric_iou_val(
                seg_prediction.view(-1),
                batch['birdview_label'][:, :self.rf].view(-1)
            )

        if self.cfg.EVAL.RGB_SUPERVISION:
            self.ssim_metric_val.add_batch(
                prediction=output[f'rgb_1'].detach(),
                target=batch[f'rgb_label_1'][:, :self.rf],
            )

        if self.cfg.LIDAR_RE.ENABLED:
            lidar_target = batch['range_view_label_1'][:, :self.rf]
            lidar_pred = output['lidar_reconstruction_1'].detach()
            pcd_target = lidar_target.detach().cpu().permute(0, 1, 3, 4, 2).flatten(2, 3).flatten(0, 1) * self.cfg.LIDAR_RE.SCALE
            pcd_pred = lidar_pred.detach().cpu().permute(0, 1, 3, 4, 2).flatten(2, 3).flatten(0, 1) * self.cfg.LIDAR_RE.SCALE
            index = np.random.randint(0, pcd_target.size(-2), 1000)
            self.cd_metric_val.add_batch(pcd_pred[:, index, :-1], pcd_target[:, index, :-1])

        if self.cfg.VOXEL_SEG.ENABLED:
            self.compute_ssc_metrics(batch, output, self.ssc_metrics_val)

        self.logging_and_visualisation(batch, output, output_imagine, loss, batch_idx, prefix='val')

        return {'val_loss': self.loss_reducing(loss)}

    def logging_and_visualisation(self, batch, output, output_imagine, loss, batch_idx, prefix='train'):
        # Logging
        self.log('-global_step', torch.tensor(-self.global_step, dtype=torch.float32))
        for key, value in loss.items():
            self.log(f'{prefix}_{key}', value)

        # if self.cfg.EVAL.RGB_SUPERVISION:
        #     ssim_value = self.ssim_metric(
        #         prediction=output[f'rgb_1'].detach(),
        #         target=batch[f'rgb_label_1'][:, :self.rf],
        #     )
        #     self.log(f'{prefix}_ssim', ssim_value)

        #  Visualisation
        if prefix == 'train':
            visualisation_criteria = (self.global_step % self.cfg.LOG_VIDEO_INTERVAL == 0) \
                                     & (self.global_step != self.vis_step)
            self.vis_step = self.global_step
        else:
            visualisation_criteria = batch_idx == 0
        if visualisation_criteria:
            self.visualise(batch, output, output_imagine, batch_idx, prefix=prefix)

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

        if self.cfg.EVAL.RGB_SUPERVISION:
            self.log(f'val_ssim', self.ssim_metric_val.get_stat())
            self.ssim_metric_val.reset()

        if self.cfg.LIDAR_RE.ENABLED:
            self.log(f'val_chamfer_distance', self.cd_metric_val.get_stat())
            self.cd_metric_val.reset()

        if self.cfg.VOXEL_SEG.ENABLED:
            # class_names_voxel = ['Background', 'Road', 'RoadLines', 'Sidewalk', 'Vehicle',
            #                      'Pedestrian', 'TrafficSign', 'TrafficLight', 'Others']
            class_names_voxel = list(VOXEL_LABEL.values())
            metric_list = [("train", self.ssc_metrics_train), ("val", self.ssc_metrics_val)]

            for prefix, metric in metric_list:
                stats = metric.get_stats()
                for i, class_name in enumerate(class_names_voxel):
                    self.log(f'{prefix}_Voxel_{class_name}_SemIoU', stats['iou_ssc'][i])
                self.log(f'{prefix}_Voxel_mIoU', stats["iou_ssc_mean"])
                self.log(f'{prefix}_Voxel_IoU', stats["iou"])
                self.log(f'{prefix}_Voxel_Precision', stats["precision"])
                self.log(f'{prefix}_Voxel_Recall', stats["recall"])
                metric.reset()

    def visualise(self, batch, output, output_imagine, batch_idx, prefix='train', writer=None):
        writer = writer if writer else self.logger.experiment
        s = self.cfg.RECEPTIVE_FIELD
        f = self.cfg.FUTURE_HORIZON

        name = f'{prefix}_outputs'
        if prefix == 'val' or prefix == 'pred':
            name = name + f'_{batch_idx}'
        # global_step = batch_idx if prefix == 'pred' else self.global_step
        global_step = self.global_step

        if self.cfg.SEMANTIC_SEG.ENABLED:

            # target = batch['birdview_label'][:, :, 0]
            # pred = torch.argmax(output['bev_segmentation_1'].detach(), dim=-3)

            # colours = torch.tensor(BIRDVIEW_COLOURS, dtype=torch.uint8, device=pred.device)

            # target = colours[target]
            # pred = colours[pred]

            # # Move channel to third position
            # target = target.permute(0, 1, 4, 2, 3)
            # pred = pred.permute(0, 1, 4, 2, 3)

            # visualisation_video = torch.cat([target, pred], dim=-1).detach()

            # # Rotate for visualisation
            # visualisation_video = torch.rot90(visualisation_video, k=1, dims=[3, 4])

            # name = f'{prefix}_outputs'
            # if prefix == 'val':
            #     name = name + f'_{batch_idx}'
            # self.logger.experiment.add_video(name, visualisation_video, global_step=self.global_step, fps=2)

            target = batch['birdview_label'][:, :, 0].cpu()
            pred = torch.argmax(output['bev_segmentation_1'].detach().cpu(), dim=-3)
            bev_imagines = []
            for imagine in output_imagine:
                bev_imagines.append(torch.argmax(imagine['bev_segmentation_1'].detach().cpu(), dim=-3))

            colours = torch.tensor(BIRDVIEW_COLOURS, dtype=torch.uint8, device=pred.device) / 255.0

            target = colours[target]
            # pred = colours[pred]

            # Move channel to third position
            target = F.pad(target.permute(0, 1, 4, 2, 3), [2, 2, 2, 2], 'constant', 0.8)
            # pred = F.pad(pred.permute(0, 1, 4, 2, 3), [2, 2, 2, 2], 'constant', 0.8)
            preds = []
            for i, bev_imagine in enumerate(bev_imagines):
                bev_receptive = pred if i == 0 else torch.zeros_like(pred)
                p_i = torch.cat([bev_receptive, bev_imagine], dim=1)
                p_i = colours[p_i]
                p_i = F.pad(p_i.permute(0, 1, 4, 2, 3), [2, 2, 2, 2], 'constant', 0.8)
                preds.append(p_i)

            bev = torch.cat([*preds[::-1], target], dim=-1).detach()
            bev = torch.rot90(bev, k=1, dims=[3, 4])

            b, _, c, h, w = bev.size()

            visualisation_bev = []
            for step in range(s + f):
                if step == s:
                    visualisation_bev.append(torch.ones(b, c, h, int(w / 4), device=pred.device))
                visualisation_bev.append(bev[:, step])
            visualisation_bev = torch.cat(visualisation_bev, dim=-1).detach()

            name_ = f'{name}_bev'
            writer.add_images(name_, visualisation_bev, global_step=global_step)

        if self.cfg.EVAL.RGB_SUPERVISION:
            # rgb_target = batch['rgb_label_1']
            # rgb_pred = output['rgb_1'].detach()

            # visualisation_rgb = torch.cat([rgb_pred, rgb_target], dim=-2).detach()
            # name_ = f'{name}_rgb'
            # writer.add_video(name_, visualisation_rgb, global_step=global_step, fps=2)

            rgb_target = batch['rgb_label_1'].cpu()
            rgb_pred = output['rgb_1'].detach().cpu()
            rgb_imagines = []
            for imagine in output_imagine:
                rgb_imagines.append(imagine['rgb_1'].detach().cpu())

            b, _, c, h, w = rgb_target.size()

            rgb_preds = []
            for i, rgb_imagine in enumerate(rgb_imagines):
                rgb_receptive = rgb_pred if i == 0 else torch.ones_like(rgb_pred)
                pred_imagine = torch.cat([rgb_receptive, rgb_imagine], dim=1)
                rgb_preds.append(F.pad(pred_imagine, [5, 5, 5, 5], 'constant', 0.8))

            rgb_target = F.pad(rgb_target, [5, 5, 5, 5], 'constant', 0.8)
            # rgb_pred = F.pad(rgb_pred, [5, 5, 5, 5], 'constant', 0.8)

            acc = batch['throttle_brake']
            steer = batch['steering']

            acc_bar = np.ones((b, s + f, int(h / 4), w + 10, c)).astype(np.uint8) * 255
            steer_bar = np.ones((b, s + f, int(h / 4), w + 10, c)).astype(np.uint8) * 255

            red = np.array([200, 0, 0])[None, None]
            green = np.array([0, 200, 0])[None, None]
            blue = np.array([0, 0, 200])[None, None]
            mid = int(w / 2) + 5

            for b_idx in range(b):
                for step in range(s + f):
                    if acc[b_idx, step] >= 0:
                        acc_bar[b_idx, step, 5: -5, mid: mid + int(w / 2 * acc[b_idx, step]), :] = green
                        cv2.putText(acc_bar[b_idx, step], f'{acc[b_idx, step, 0]:.5f}', (mid - 220, int(h / 8) + 15),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                    else:
                        acc_bar[b_idx, step, 5: -5, mid + int(w / 2 * acc[b_idx, step]): mid, :] = red
                        cv2.putText(acc_bar[b_idx, step], f'{acc[b_idx, step, 0]:.5f}', (mid + 10, int(h / 8) + 15),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                    if steer[b_idx, step] >= 0:
                        steer_bar[b_idx, step, 5: -5, mid: mid + int(w / 2 * steer[b_idx, step]), :] = blue
                        cv2.putText(steer_bar[b_idx, step], f'{steer[b_idx, step, 0]:.5f}',
                                    (mid - 220, int(h / 8) + 15),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                    else:
                        steer_bar[b_idx, step, 5: -5, mid + int(w / 2 * steer[b_idx, step]): mid, :] = blue
                        cv2.putText(steer_bar[b_idx, step], f'{steer[b_idx, step, 0]:.5f}', (mid + 10, int(h / 8) + 15),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
            acc_bar = torch.tensor(acc_bar.transpose((0, 1, 4, 2, 3)),
                                   dtype=torch.float, device=rgb_pred.device) / 255.0
            steer_bar = torch.tensor(steer_bar.transpose((0, 1, 4, 2, 3)),
                                     dtype=torch.float, device=rgb_pred.device) / 255.0

            rgb = torch.cat([acc_bar, steer_bar, rgb_target, *rgb_preds], dim=-2)
            visualisation_rgb = []
            for step in range(s + f):
                if step == s:
                    visualisation_rgb.append(torch.ones(b, c, rgb.size(-2), int(w / 4), device=rgb_pred.device))
                visualisation_rgb.append(rgb[:, step, ...])
            visualisation_rgb = torch.cat(visualisation_rgb, dim=-1).detach()

            name_ = f'{name}_rgb'
            writer.add_images(name_, visualisation_rgb, global_step=global_step)

            ###################################
            flows = []
            rgb_target_np = (rgb_target.detach().cpu().numpy().transpose(0, 1, 3, 4, 2) * 255).astype(np.uint8)
            rgb_preds_np = [(rgb_pred_.detach().cpu().numpy().transpose(0, 1, 3, 4, 2) * 255).astype(np.uint8)
                           for rgb_pred_ in rgb_preds]
            for bs in range(rgb.size(0)):
                flows.append(list())
                for step in range(1, rgb.size(1)):
                    img1_target = rgb_target_np[bs, step-1][5: -5, 5: -5]
                    img2_target = rgb_target_np[bs, step][5: -5, 5: -5]
                    flow_target = self.get_color_coded_flow(img1_target, img2_target)
                    flow_target = F.pad(flow_target, [5, 5, 5, 5], 'constant', 0.8)

                    flow_preds = []
                    for i, rgb_pred_np in enumerate(rgb_preds_np):
                        img1_pred = rgb_pred_np[bs, step-1][5: -5, 5: -5]
                        if i == s:
                            img1_pred = rgb_preds_np[0][bs, step-1][5: -5, 5: -5]
                        img2_pred = rgb_pred_np[bs, step][5: -5, 5: -5]
                        flow_pred = self.get_color_coded_flow(img1_pred, img2_pred)
                        flow_pred = F.pad(flow_pred, [5, 5, 5, 5], 'constant', 0.8)
                        flow_preds.append(flow_pred)

                    flows[bs].append(torch.cat([flow_target, *flow_preds], dim=1))

            visualisation_flow = torch.stack([torch.cat(flow, dim=-1) for flow in flows], dim=0)
            name_ = f'{name}_flow'
            writer.add_images(name_, visualisation_flow, global_step=global_step)

        if self.cfg.LIDAR_RE.ENABLED:
            lidar_target = batch['range_view_label_1'].cpu()
            lidar_pred = output['lidar_reconstruction_1'].detach().cpu()
            # lidar_imagine = output_imagine[0]['lidar_reconstruction_1'].detach()
            lidar_imagines = [imagine['lidar_reconstruction_1'].detach().cpu() for imagine in output_imagine]

            lidar_pred_imagine = torch.cat([lidar_pred, lidar_imagines[0]], dim=1)
            visualisation_lidar = torch.cat(
                [lidar_pred_imagine[:, :, -1, :, :], lidar_target[:, :, -1, :, :]],
                dim=-2).detach().unsqueeze(-3)
            name_ = f'{name}_lidar'
            writer.add_video(name_, visualisation_lidar, global_step=global_step, fps=2)

            pcd_image_target, pcd_target = self.pcd_xy_image(lidar_target)
            pcd_image_target = F.pad(pcd_image_target, [2, 2, 2, 2], 'constant', 0.2)

            pcd_image_pred, pcd_pred = self.pcd_xy_image(lidar_pred)

            pcd_image_preds = []
            for i, lidar_imagine in enumerate(lidar_imagines):
                pcd_image_imagine, pcd_imagine = self.pcd_xy_image(lidar_imagine)
                pcd_receptive = pcd_image_pred if i == 0 else torch.ones_like(pcd_image_pred)
                pcd_pred_imagine = torch.cat([pcd_receptive, pcd_image_imagine], dim=1)
                pcd_image_preds.append(F.pad(pcd_pred_imagine, [2, 2, 2, 2], 'constant', 0.2))

            pcd_image = torch.cat([pcd_image_target, *pcd_image_preds], dim=-2)
            b, _, c, h, w = pcd_image.size()

            visualisation_pcd = []
            for step in range(s + f):
                if step == s:
                    visualisation_pcd.append(torch.ones(b, c, h, int(w / 4), device=pcd_image.device))
                visualisation_pcd.append(pcd_image[:, step])
            visualisation_pcd = torch.cat(visualisation_pcd, dim=-1).detach()

            name_ = f'{name}_pcd_xy'
            writer.add_images(name_, visualisation_pcd, global_step=global_step)

            # if self.cml_logger is not None:
            #     name_ = f'{name}_pcd'
            #     self.cml_logger.report_scatter3d(title=f'{name_}_target', series=prefix, scatter=pcd_target,
            #                                      iteration=self.global_step, mode='markers',
            #                                      extra_layout={'marker': {'size': 1}})
            #     self.cml_logger.report_scatter3d(title=f'{name_}_target_d', series=prefix, scatter=pcd_target0,
            #                                      iteration=self.global_step, mode='markers',
            #                                      extra_layout={'marker': {'size': 1}})
            #     self.cml_logger.report_scatter3d(title=f'{name_}_pred', series=prefix, scatter=pcd_pred,
            #                                      iteration=self.global_step, mode='markers',
            #                                      extra_layout={'marker': {'size': 1}})
            #     self.cml_logger.report_scatter3d(title=f'{name_}_pred_d', series=prefix, scatter=pcd_pred0,
            #                                      iteration=self.global_step, mode='markers',
            #                                      extra_layout={'marker': {'size': 1}})

            # writer.add_mesh(f'{name_}_target', vertices=pcd_target)
            # writer.add_mesh(f'{name_}_target_d', vertices=pcd_target0[None])
            # writer.add_mesh(f'{name_}_pred', vertices=pcd_pred1)
            # writer.add_mesh(f'{name_}_pred_d', vertices=pcd_pred0[None])

        if self.cfg.LIDAR_SEG.ENABLED:
            lidar_seg_target = batch['range_view_seg_label_1'][:, :, 0].cpu()
            lidar_seg_pred = torch.argmax(output['lidar_segmentation_1'].detach().cpu(), dim=-3)
            lidar_seg_imagine = torch.argmax(output_imagine[0]['lidar_segmentation_1'].detach().cpu(), dim=-3)
            lidar_seg_pred = torch.cat([lidar_seg_pred, lidar_seg_imagine], dim=1)

            colours = torch.tensor(VOXEL_COLOURS, dtype=torch.uint8, device=lidar_seg_pred.device)
            lidar_seg_target = colours[lidar_seg_target]
            lidar_seg_pred = colours[lidar_seg_pred]

            lidar_seg_target = lidar_seg_target.permute(0, 1, 4, 2, 3)
            lidar_seg_pred = lidar_seg_pred.permute(0, 1, 4, 2, 3)

            visualisation_lidar_seg = torch.cat([lidar_seg_pred, lidar_seg_target], dim=-2).detach()
            name_ = f'{name}_lidar_seg'
            writer.add_video(name_, visualisation_lidar_seg, global_step=global_step, fps=2)

        if self.cfg.SEMANTIC_IMAGE.ENABLED:
            sem_target = batch['semantic_image_label_1'][:, :, 0].cpu()
            sem_pred = torch.argmax(output['semantic_image_1'].detach().cpu(), dim=-3)
            sem_imagine = torch.argmax(output_imagine[0]['semantic_image_1'].detach().cpu(), dim=-3)
            sem_pred = torch.cat([sem_pred, sem_imagine], dim=1)

            colours = torch.tensor(VOXEL_COLOURS, dtype=torch.uint8, device=sem_pred.device)
            sem_target = colours[sem_target]
            sem_pred = colours[sem_pred]

            sem_target = sem_target.permute(0, 1, 4, 2, 3)
            sem_pred = sem_pred.permute(0, 1, 4, 2, 3)

            visualisation_sem_image = torch.cat([sem_pred, sem_target], dim=-2).detach()
            name_ = f'{name}_sem_image'
            writer.add_video(name_, visualisation_sem_image, global_step=global_step, fps=2)

        if self.cfg.DEPTH.ENABLED:
            depth_target = batch['depth_label_1'].cpu()
            depth_pred = output['depth_1'].detach().cpu()
            depth_imagine = output_imagine[0]['depth_1'].detach().cpu()
            depth_pred = torch.cat([depth_pred, depth_imagine], dim=1)

            visualisation_depth = torch.cat([depth_pred, depth_target], dim=-2).detach()
            name_ = f'{name}_depth'
            writer.add_video(name_, visualisation_depth, global_step=global_step, fps=2)

        if self.cfg.VOXEL_SEG.ENABLED:
            voxel_target = batch['voxel_label_1'][0, 0, 0].cpu().numpy()
            voxel_pred = torch.argmax(output['voxel_1'].detach(), dim=-4).cpu().numpy()[0, 0]
            voxel_imagine_target = batch['voxel_label_1'][0, self.rf, 0].cpu().numpy()
            voxel_imagine_pred = torch.argmax(output_imagine[0]['voxel_1'].detach(), dim=-4).cpu().numpy()[0, 0]
            colours = np.asarray(VOXEL_COLOURS, dtype=float) / 255.0
            voxel_color_target = colours[voxel_target]
            voxel_color_pred = colours[voxel_pred]
            voxel_color_imagine_target = colours[voxel_imagine_target]
            voxel_color_imagine_pred = colours[voxel_imagine_pred]
            name_ = f'{name}_voxel'
            self.write_voxel_figure(voxel_target, voxel_color_target, f'{name_}_target', global_step, writer)
            self.write_voxel_figure(voxel_pred, voxel_color_pred, f'{name_}_pred', global_step, writer)
            self.write_voxel_figure(voxel_target, voxel_color_imagine_target, f'{name_}_target', global_step, writer)
            self.write_voxel_figure(voxel_pred, voxel_color_imagine_pred, f'{name_}_pred', global_step, writer)

        if self.cfg.MODEL.ROUTE.ENABLED:
            route_map = batch['route_map'].cpu()
            route_map = F.pad(route_map, [2, 2, 2, 2], 'constant', 0.8)

            b, _, c, h, w = route_map.size()

            visualisation_route = []
            for step in range(s + f):
                if step == s:
                    visualisation_route.append(torch.ones(b, c, h, int(w / 4), device=route_map.device))
                visualisation_route.append(route_map[:, step])
            visualisation_route = torch.cat(visualisation_route, dim=-1).detach()

            name_ = f'{name}_input_route_map'
            writer.add_images(name_, visualisation_route, global_step=global_step)

    def write_voxel_figure(self, voxel, voxel_color, name, global_step, writer):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.voxels(voxel, facecolors=voxel_color, shade=False)
        ax.view_init(elev=60, azim=165)
        ax.set_axis_off()
        writer.add_figure(name, fig, global_step=global_step)

    def pcd_xy_image(self, lidar):
        image_size = np.array([256, 256])
        lidar_range = 50

        pcd = lidar.cpu().detach().numpy().transpose(0, 1, 3, 4, 2) * self.cfg.LIDAR_RE.SCALE
        # pcd_target = pcd_target[..., :-1].flatten(1, 2)
        # pcd_target = pcd_target[pcd_target[..., -1] > 0][..., :-1]
        pcd0 = self.pcd.restore_pcd_coor(lidar[:, :, -1].cpu().numpy() * self.cfg.LIDAR_RE.SCALE)
        pcd_xy = -pcd[..., :2]
        pcd_xy *= min(image_size) / (2 * lidar_range)
        pcd_xy += 0.5 * image_size.reshape((1, 1, 1, 1, -1))
        valid = pcd[..., -1] > 0

        b, s, _, _, _ = pcd.shape
        pcd_xy_image = np.zeros((b, s, *image_size, 3))

        for i in range(b):
            for j in range(s):
                hw = pcd_xy[i, j][valid[i, j]]
                hw = hw[(0 < hw[:, 0]) & (hw[:, 0] < image_size[0]) &
                        (0 < hw[:, 1]) & (hw[:, 1] < image_size[1])]
                hw = np.fabs(hw)
                hw = hw.astype(np.int32)
                pcd_xy_image[i, j][tuple(hw.T)] = (1.0, 1.0, 1.0)

        return torch.tensor(pcd_xy_image.transpose((0, 1, 4, 2, 3)), device=lidar.device), pcd
    
    def get_color_coded_flow(self, img1, img2):
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 2] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        color_coded_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return torch.tensor(color_coded_flow.transpose(2, 0, 1), dtype=torch.float) / 255.0

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

    def on_test_epoch_end(self):
        class_names = ['Background', 'Road', 'Lane marking', 'Vehicle', 'Pedestrian', 'Green light', 'Yellow light',
                       'Red light and stop sign']
        if self.cfg.SEMANTIC_SEG.ENABLED:
            scores = self.metric_iou_test.compute()
            for key, value in zip(class_names, scores):
                self.log('pred_iou_' + key, value)
            self.log('pred_mean_iou', torch.mean(scores))
            self.metric_iou_test.reset()

        if self.cfg.EVAL.RGB_SUPERVISION:
            self.log(f'pred_ssim', self.ssim_metric_test.get_stat())
            self.ssim_metric_test.reset()

        if self.cfg.LIDAR_RE.ENABLED:
            self.log(f'pred_chamfer_distance', self.cd_metric_test.get_stat())
            self.cd_metric_test.reset()

        if self.cfg.VOXEL_SEG.ENABLED:
            # class_names_voxel = ['Background', 'Road', 'RoadLines', 'Sidewalk', 'Vehicle',
            #                      'Pedestrian', 'TrafficSign', 'TrafficLight', 'Others']
            class_names_voxel = list(VOXEL_LABEL.values())
            metric_list = [("pred", self.ssc_metrics_test)]

            for prefix, metric in metric_list:
                stats = metric.get_stats()
                for i, class_name in enumerate(class_names_voxel):
                    self.log(f'{prefix}_Voxel_{class_name}_SemIoU', stats['iou_ssc'][i])
                self.log(f'{prefix}_Voxel_mIoU', stats["iou_ssc_mean"])
                self.log(f'{prefix}_Voxel_IoU', stats["iou"])
                self.log(f'{prefix}_Voxel_Precision', stats["precision"])
                self.log(f'{prefix}_Voxel_Recall', stats["recall"])
                metric.reset()

    def test_step(self, batch, batch_idx, predict_action=False):
        self.train()
        for module in self.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        with torch.no_grad():
            output, output_imagine = self.forward(batch, predict_action=predict_action)
        self.eval()

        if self.cfg.EVAL.RGB_SUPERVISION:
            self.ssim_metric_test.add_batch(
                prediction=output[f'rgb_1'].detach(),
                target=batch[f'rgb_label_1'][:, :self.rf],
            )

        if self.cfg.SEMANTIC_SEG.ENABLED:
            seg_prediction = output['bev_segmentation_1'].detach()
            seg_prediction = torch.argmax(seg_prediction, dim=2)
            self.metric_iou_test(
                seg_prediction.view(-1),
                batch['birdview_label'][:, :self.rf].view(-1)
            )

        if self.cfg.LIDAR_RE.ENABLED:
            lidar_target = batch['range_view_label_1'][:, :self.rf]
            lidar_pred = output['lidar_reconstruction_1'].detach()
            pcd_target = lidar_target.detach().cpu().permute(0, 1, 3, 4, 2).flatten(2, 3).flatten(0, 1) * self.cfg.LIDAR_RE.SCALE
            pcd_pred = lidar_pred.detach().cpu().permute(0, 1, 3, 4, 2).flatten(2, 3).flatten(0, 1) * self.cfg.LIDAR_RE.SCALE
            index = np.random.randint(0, pcd_target.size(-2), 1000)
            self.cd_metric_test.add_batch(pcd_pred[:, index, :-1], pcd_target[:, index, :-1])

        if self.cfg.VOXEL_SEG.ENABLED:
            self.compute_ssc_metrics(batch, output, self.ssc_metrics_test)

        self.visualise(batch, output, output_imagine, batch_idx, prefix='pred')
        return output, output_imagine

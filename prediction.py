import os
import socket
import time
from tqdm import tqdm

import torch
# from torch.utils.tensorboard.writer import SummaryWriter
import lightning.pytorch as pl
import numpy as np

from muvo.config import get_parser, get_cfg
from muvo.data.dataset import DataModule
from muvo.trainer import WorldModelTrainer
from lightning.pytorch.callbacks import ModelSummary

from clearml import Task, Dataset, Model


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    task = Task.init(project_name=cfg.CML_PROJECT, task_name=cfg.CML_TASK, task_type=cfg.CML_TYPE, tags=cfg.TAG)
    task.connect(cfg)
    cml_logger = task.get_logger()
    #
    # dataset_root = Dataset.get(dataset_project=cfg.CML_PROJECT,
    #                            dataset_name=cfg.CML_DATASET,
    #                            ).get_local_copy()

    # data = DataModule(cfg, dataset_root=dataset_root)
    data = DataModule(cfg)
    data.setup()

    input_model = Model(model_id='').get_local_copy() if cfg.PRETRAINED.CML_MODEL else None
    model = WorldModelTrainer(cfg.convert_to_dict(), pretrained_path=input_model)
    # model.get_cml_logger(cml_logger)

    save_dir = os.path.join(
        cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    )
    # writer = SummaryWriter(log_dir=save_dir)

    dataloader = data.test_dataloader()[2]

    pbar = tqdm(total=len(dataloader),  desc='Prediction')
    model.cuda()

    model.eval()
    model.model.train()
    for module in model.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()

    n_prediction_samples = model.cfg.PREDICTION.N_SAMPLES

    for i, batch in enumerate(dataloader):
        output_imagines = []
        ims = []
        batch = {key: value.cuda() for key, value in batch.items()}
        with torch.no_grad():
            batch = model.preprocess(batch)
            batch_rf = {key: value[:, :model.rf] for key, value in batch.items()}  # dim (b, s, 512)
            # batch_fh = {key: value[:, model.rf:] for key, value in batch.items()}  # dim (b, s, 512)
            output, state_dict = model.model.forward(batch_rf, deployment=False)

            state_imagine = {'hidden_state': state_dict['posterior']['hidden_state'][:, -1],
                             'sample': state_dict['posterior']['sample'][:, -1],
                             'throttle_brake': batch['throttle_brake'][:, model.rf:],
                             'steering': batch['steering'][:, model.rf:]}
            for _ in range(n_prediction_samples):
                output_imagine = model.model.imagine(state_imagine, predict_action=False, future_horizon=model.fh)
                output_imagines.append(output_imagine)
                voxel_im = torch.where(torch.argmax(output_imagine['voxel_1'].detach(), dim=-4).cpu() != 0)
                voxel_im = torch.stack(voxel_im).transpose(0, 1).numpy()
                ims.append({'rgb_re': output_imagine['rgb_1'].detach().cpu().numpy(),
                            'pcd_re': output_imagine['lidar_reconstruction_1'].detach().cpu().numpy(),
                            'voxel_re': voxel_im,
                            })

            model.visualise(batch, output, output_imagines, batch_idx=i, prefix='pred', writer=writer)
            voxel_label = torch.where(batch['voxel_label_1'].cpu() != 0)
            voxel_label = torch.stack(voxel_label).transpose(0, 1).numpy()
            voxel_re = torch.where(torch.argmax(output['voxel_1'].detach(), dim=-4).cpu() != 0)
            voxel_re = torch.stack(voxel_re).transpose(0, 1).numpy()
            gt = {'rgb_label': batch['rgb_label_1'].cpu().numpy(),
                  'throttle_brake': batch['throttle_brake'].cpu().numpy(),
                  'steering': batch['steering'].cpu().numpy(),
                  'pcd_label': batch['range_view_label_1'].cpu().numpy(),
                  'voxel_label': voxel_label,
                  }

            re = {'rgb_re': output['rgb_1'].detach().cpu().numpy(),
                  'pcd_re': output['lidar_reconstruction_1'].detach().cpu().numpy(),
                  'voxel_re': voxel_re,
                  }
            upload_data = {'gt': gt, 're': re, 'ims': ims}
            task.upload_artifact(f'data_{i}', np.array(upload_data))
        pbar.update(1)
            

if __name__ == '__main__':
    main()

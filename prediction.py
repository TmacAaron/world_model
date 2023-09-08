import os
import socket
import time
from tqdm import tqdm

import torch
from torch.utils.tensorboard.writer import SummaryWriter
import lightning.pytorch as pl

from mile.config import get_parser, get_cfg
from mile.data.dataset import DataModule
from mile.trainer import WorldModelTrainer

from clearml import Task, Dataset, Model


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    task = Task.init(project_name=cfg.CML_PROJECT, task_name=cfg.CML_TASK, task_type=cfg.CML_TYPE, tags=cfg.TAG)
    task.connect(cfg)
    # cml_logger = task.get_logger()
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
    writer = SummaryWriter(log_dir=save_dir)

    dataloader = data.predict_dataloader()

    pbar = tqdm(total=len(dataloader),  desc='Prediction')
    model.cuda()

    for i, data in enumerate(dataloader):
        data = {key: value.cuda() for key, value in data.items()}
        with torch.no_grad():
            output, output_imagine = model.forward(data)
            model.visualise(data, output, output_imagine, batch_idx=i, prefix='pred', writer=writer)
        pbar.update(1)


if __name__ == '__main__':
    main()

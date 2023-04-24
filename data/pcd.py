import numpy as np
from data_preprocessing import *
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import re
from tqdm import tqdm
# from clearml import Task
import logging
from multiprocessing import Pool, RLock

log = logging.getLogger(__name__)


def process_one_file(data_path, transition, task_idx, all_task, position, is_semantic=True):
    if is_semantic:
        save_path = data_path.joinpath('points_semantic')
        file = data_path.joinpath('point_clouds_semantic.npy')
    else:
        save_path = data_path.joinpath('points')
        file = data_path.joinpath('point_clouds.npy')
    if not save_path.exists():
        save_path.mkdir()
    try:
        pcd_list = load_lidar(file)
        pbar = tqdm(total=len(pcd_list), desc=f'{task_idx + 1:04} / {all_task:04}',
                    position=position, postfix='semantic' if is_semantic else 'points')
        for name, lidar_unprocessed in pcd_list.items():
            lidar_processed = process_pcd(lidar_unprocessed, transition)
            np.save(f'{save_path}/{name}.npy', lidar_processed)
            pbar.update()
    except Exception as e:
        log.error(f'{e}')

    log.info(f"Saved processed points clouds in {save_path}.")


def process_dir(data_path, cfg, task_idx, all_task):
    log.info(f'Process points clouds in {data_path}.')
    process_one_file(data_path, cfg.lidar_transition, task_idx, all_task, task_idx % cfg.n_process)
    process_one_file(data_path, cfg.lidar_transition, task_idx, all_task, task_idx % cfg.n_process, False)


@hydra.main(config_path='./', config_name='data_preprocess')
def main(cfg: DictConfig):
    # task = Task.init(project_name=cfg.cml_project, task_name=cfg.cml_task_name, task_type=cfg.cml_task_type,
    #                  tags=cfg.cml_tags)
    # task.connect(cfg)
    # cml_logger = task.get_logger()

    root_path = Path(cfg.root)
    data_paths = sorted([p for p in root_path.glob('**/Town*/*/') if p.is_dir()])

    if not root_path.exists() or len(data_paths) == 0:
        print('Root Path does not EXSIT or there are NO LEGAL files!!!')
        return

    log.info(f'{len(data_paths)} points sequences will be proceed.')

    p = Pool(cfg.n_process, initializer=tqdm.set_lock, initargs=(RLock(),))
    for i, path in enumerate(data_paths):
        p.apply_async(func=process_dir, args=(path, cfg, i, len(data_paths)))
    p.close()
    p.join()

    log.info("Finished!")


if __name__ == '__main__':
    main()

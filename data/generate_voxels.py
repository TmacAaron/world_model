import numpy as np
from data_preprocessing import *
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import scipy.sparse as sp
import re
from tqdm import tqdm
from clearml import Task
import logging

log = logging.getLogger(__name__)


@hydra.main(config_path='./', config_name='data_preprocess')
def main(cfg: DictConfig):
    task = Task.init(project_name=cfg.cml_project, task_name=cfg.cml_task_name, task_type=cfg.cml_task_type,
                     tags=cfg.cml_tags)
    task.connect(cfg)
    cml_logger = task.get_logger()

    root_path = Path(cfg.root)
    data_paths = [p for p in root_path.glob('**/depth_semantic') if p.is_dir()]
    for i, p in enumerate(data_paths):
        log.info(f'Converting Depth Image to Voxels in {p}.')
        save_path = p.parent
        # if not save_path.exists():
        #     save_path.mkdir()
        file_list = sorted([str(f) for f in p.glob('*.png')])
        data_dict = {}
        voxels_dict = {}
        for file in tqdm(file_list[:250], desc=f'{i+1:04} / {len(data_paths):04}'):
            depth, semantic, _ = read_img(file)
            points_list, sem_list = get_all_points(
                depth, semantic, fov=cfg.fov, size=cfg.size, offset=cfg.offset, mask_ego=cfg.mask_ego)
            voxel_points, semantics, points_filtered = voxel_filter(points_list, sem_list, cfg.voxel_size, cfg.center)
            data = np.concatenate([voxel_points, semantics[:, None]], axis=1)
            voxels = np.zeros(shape=(2 * np.asarray(cfg.center) / cfg.voxel_size).astype(int), dtype=np.uint8)
            voxels[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]] = semantics
            name = re.match(r'.*/.*_(\d{9})\.png', file).group(1)
            # np.savez_compressed(f"{save_path}/{name}.npz", data=data)
            coo_voxels = sp.coo_matrix(voxels.reshape(voxels.shape[0], -1))
            # np.savez_compressed(f"{save_path}/v{name}.npz", data=coo_voxels)
            voxels_dict[name] = coo_voxels
            data_dict[name] = data
        np.savez_compressed(f"{save_path}/voxels.npz", voxels=voxels_dict, points=data_dict)
        log.info(f"Saved Voxels Data in {save_path}/voxels.npz.")


if __name__ == '__main__':
    main()

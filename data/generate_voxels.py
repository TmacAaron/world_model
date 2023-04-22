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
from multiprocessing import Pool, RLock, Pipe
from threading import Thread

log = logging.getLogger(__name__)


def progress_bar_total(parent, total_len, desc):
    desc = desc if desc else "Main"
    pbar_main = tqdm(total=total_len, desc=desc, position=0)
    nums = 0
    while True:
        msg = parent.recv()[0]
        if msg is not None:
            pbar_main.update()
            nums += 1
        if nums == total_len:
            break
    pbar_main.close()


def voxelize_dir(data_path, cfg, task_idx, all_task, pipe):
    log.info(f'Converting Depth Image to Voxels in {data_path}.')
    save_path = data_path.parent
    # if not save_path.exists():
    #     save_path.mkdir()
    file_list = sorted([str(f) for f in data_path.glob('*.png')])
    data_dict = {}
    voxels_dict = {}
    for file in tqdm(file_list, desc=f'{task_idx + 1:04} / {all_task:04}', position=task_idx % cfg.n_process + 1):
        depth, semantic, _ = read_img(file)
        points_list, sem_list = get_all_points(
            depth, semantic, fov=cfg.fov, size=cfg.size, offset=cfg.offset, mask_ego=cfg.mask_ego)
        voxel_points, semantics = voxel_filter(points_list, sem_list, cfg.voxel_size, cfg.center)
        data = np.concatenate([voxel_points, semantics[:, None]], axis=1)
        voxels = np.zeros(shape=(2 * np.asarray(cfg.center) / cfg.voxel_size).astype(int), dtype=np.uint8)
        voxels[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]] = semantics
        name = re.match(r'.*/.*_(\d{9})\.png', file).group(1)
        # np.savez_compressed(f"{save_path}/{name}.npz", data=data)
        coo_voxels = sp.coo_matrix(voxels.reshape(voxels.shape[0], -1))
        # np.savez_compressed(f"{save_path}/v{name}.npz", data=coo_voxels)
        voxels_dict[name] = coo_voxels
        data_dict[name] = data

        if pipe is not None:
            pipe.send(['x'])

    np.savez_compressed(f"{save_path}/voxels.npz", coo_voxels=voxels_dict, voxel_points=data_dict)
    log.info(f"Saved Voxels Data in {save_path}/voxels.npz.")


@hydra.main(config_path='./', config_name='data_preprocess')
def main(cfg: DictConfig):
    task = Task.init(project_name=cfg.cml_project, task_name=cfg.cml_task_name, task_type=cfg.cml_task_type,
                     tags=cfg.cml_tags)
    task.connect(cfg)
    cml_logger = task.get_logger()

    root_path = Path(cfg.root)
    data_paths = sorted([p for p in root_path.glob('**/depth_semantic') if p.is_dir()])
    n_files = len([f for f in root_path.glob('**/depth_semantic/*.png')])

    parent, child = Pipe()
    main_thread = Thread(target=progress_bar_total, args=(parent, n_files, "Total"))
    main_thread.start()
    p = Pool(cfg.n_process, initializer=tqdm.set_lock, initargs=(RLock(),))
    for i, path in enumerate(data_paths):
        p.apply_async(func=voxelize_dir, args=(path, cfg, i, len(data_paths), child))
    p.close()
    p.join()
    main_thread.join()

    log.info("Finished Voxelization!")


if __name__ == '__main__':
    main()

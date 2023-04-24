import numpy as np
# import open3d as o3d
import cv2
# import matplotlib.pyplot as plt


# LABEL = np.array([
#     (0, 0, 0, 'ego'),  # unlabeled
#     # cityscape
#     (128, 64, 128, 'road'),     # road = 1
#     (244, 35, 232, 'sidewalk'),     # sidewalk = 2
#     (70, 70, 70, 'building'),       # building = 3
#     (102, 102, 156, 'wall'),    # wall = 4
#     (190, 153, 153, 'fence'),    # fence = 5
#     (153, 153, 153, 'pole'),    # pole = 6
#     (250, 170, 30, 'traffic'),     # traffic light = 7
#     (220, 220, 0, 'traffic'),      # traffic sign = 8
#     (107, 142, 35, 'vegetation'),     # vegetation = 9
#     (152, 251, 152, 'terrain'),    # terrain = 10
#     (70, 130, 180, 'sky'),     # sky = 11
#     (220, 20, 60, 'pedestrian'),      # pedestrian = 12
#     (255, 0, 0, 'rider'),        # rider = 13
#     (0, 0, 142, 'Car'),        # Car = 14
#     (0, 0, 70, 'truck'),         # truck = 15
#     (0, 60, 100, 'bs'),       # bs = 16
#     (0, 80, 100, 'train'),       # train = 17
#     (0, 0, 230, 'motorcycle'),        # motorcycle = 18
#     (119, 11, 32, 'bicycle'),      # bicycle = 19
#     # , 'customcustom
#     (110, 190, 160, 'static'),    # static = 20
#     (170, 120, 50, 'dynamic'),     # dynamic = 21
#     (55, 90, 80, 'other'),       # other = 22
#     (45, 60, 150, 'water'),      # water = 23
#     (157, 234, 50, 'road'),     # road line = 24
#     (81, 0, 81, 'grond'),        # grond = 25
#     (150, 100, 100, 'bridge'),    # bridge = 26
#     (230, 150, 140, 'rail'),    # rail track = 27
#     (180, 165, 180, 'gard')     # gard rail = 28
# ])
LABEL = np.array([
    (255, 255, 255, 'Ego'),  # None
    (70, 70, 70, 'Building'),  # Building
    (100, 40, 40, 'Fences'),  # Fences
    (55, 90, 80, 'Other'),  # Other
    (220, 20, 60, 'Pedestrian'),  # Pedestrian
    (153, 153, 153, 'Pole'),  # Pole
    (157, 234, 50, 'RoadLines'),  # RoadLines
    (128, 64, 128, 'Road'),  # Road
    (244, 35, 232, 'Sidewalk'),  # Sidewalk
    (107, 142, 35, 'Vegetation'),  # Vegetation
    (0, 0, 142, 'Vehicle'),  # Vehicle
    (102, 102, 156, 'Wall'),  # Wall
    (220, 220, 0, 'TrafficSign'),  # TrafficSign
    (70, 130, 180, 'Sky'),  # Sky
    (81, 0, 81, 'Ground'),  # Ground
    (150, 100, 100, 'Bridge'),  # Bridge
    (230, 150, 140, 'RailTrack'),  # RailTrack
    (180, 165, 180, 'GuardRail'),  # GuardRail
    (250, 170, 30, 'TrafficLight'),  # TrafficLight
    (110, 190, 160, 'Static'),  # Static
    (170, 120, 50, 'Dynamic'),  # Dynamic
    (45, 60, 150, 'Water'),  # Water
    (145, 170, 100, 'Terrain'),  # Terrain
])
LABEL_COLORS = LABEL[:, :-1].astype(np.uint8) / 255.0
LABEL_CLASS = np.char.lower(LABEL[:, -1])


def read_img(file):
    img = cv2.imread(file, -1)
    depth_color = img[..., :-1]
    semantic = img[..., -1]
    depth = 1000 * (256 ** 2 * depth_color[..., 2] + 256 * depth_color[..., 1] + depth_color[..., 0]) / (256 ** 3 - 1)
    return depth, semantic, depth_color


def load_lidar(file):
    data = np.load(file, allow_pickle=True).item()
    return data


def depth2pcd(depth, semantic, fov):
    h, w = depth.shape
    f = w / (2.0 * np.tan(fov * np.pi / 360.0))
    cx, cy = w / 2.0, h / 2.0

    depth = depth.reshape((-1, 1))
    valid = (depth < 1000).squeeze()
    depth = depth[valid]

    g_x = np.arange(0, w)
    g_y = np.arange(0, h)
    xx, yy = np.meshgrid(g_x, g_y)
    xx, yy = xx.reshape((-1, 1))[valid], yy.reshape((-1, 1))[valid]
    x, y = (xx - cx) * depth / f, (yy - cy) * depth / f
    points_list = np.concatenate([x, y, depth], axis=1)
    sem_list = semantic.reshape((-1, 1))[valid]
    return points_list, sem_list


def get_all_points(depth, semantic, fov=90, size=(320, 320), offset=(10, 10, 10), mask_ego=True):
    h, w = size
    num_sensor = (int(depth.shape[0] / size[1]), int(depth.shape[1] / size[0]))
    points_list = []
    sem_list = []
    for i in range(num_sensor[0]):
        for j in range(num_sensor[1]):
            y0, y1 = i * h, (i + 1) * h
            x0, x1 = j * w, (j + 1) * w
            pl, sl = depth2pcd(depth[y0: y1, x0: x1], semantic[y0: y1, x0: x1], fov)
            x, y, z = offset[1] * (j - int(num_sensor[1] / 2)), offset[0] * (i - int(num_sensor[0] / 2)), -offset[2]
            pl += np.array([x, y, z])
            points_list.append(pl)
            sem_list.append(sl)
    points_list = np.concatenate(points_list, axis=0)
    sem_list = np.concatenate(sem_list, axis=0)
    points_list[:, 2] = -points_list[:, 2]
    points_list[:, :2] = -points_list[:, :2][:, ::-1]
    if mask_ego is not False:
        if type(mask_ego) is not bool:
            mask_ego = np.asarray(mask_ego)
            ego_box = np.array([-mask_ego, mask_ego])
            ego_box[0, 2] = 0
        else:
            ego_box = np.array([[-2.5, -1.1, 0], [2.5, 1.1, 2]])
        ego_idx = ((ego_box[0] < points_list) & (points_list < ego_box[1])).all(axis=1)
        sem_list[ego_idx] = 255
    return points_list, sem_list


def voxel_filter(pcd, sem, voxel_size, center, center_low=True):
    center = np.asarray(center)
    pcd_b = pcd + center
    if center_low:
        pcd_b[:, 2] -= center[2] / 2
    idx = ((0 <= pcd_b) & (pcd_b <= 2 * center)).all(axis=1)
    pcd_b, sem_b = pcd_b[idx], sem[idx]

    Dx, Dy, Dz = (2 * center) // voxel_size + 1
    # compute index for every point in a voxel
    hxyz, hmod = np.divmod(pcd_b, voxel_size)
    h = hxyz[:, 0] + hxyz[:, 1] * Dx + hxyz[:, 2] * Dx * Dy

    # h_n = np.nonzero(np.bincount(h.astype(np.int32)))
    h_idx = np.argsort(h)
    h, hxyz, sem_b, pcd_b, hmod = h[h_idx], hxyz[h_idx], sem_b[h_idx], pcd_b[h_idx], hmod[h_idx]
    h_n, indices = np.unique(h, return_index=True)
    n_f = h_n.shape[0]
    n_all = h.shape[0]
    voxels = np.zeros((n_f, 3), dtype=np.uint16)
    semantics = np.zeros((n_f, ), dtype=np.uint8)
    # points_f = np.zeros((n_f, 3))
    road_idx = np.where(LABEL_CLASS == 'roadlines')[0][0]
    # voxels = []
    # semantics = []
    # points_f = []
    for i in range(n_f):
        # idx_ = (h == h_n[i])
        idx_ = np.arange(indices[i], indices[i+1]) if i < n_f - 1 else np.arange(indices[i], n_all)
        dis = np.sum(hmod[idx_] ** 2, axis=1)
        semantic = sem_b[idx_][np.argmin(dis)] if not np.isin(sem_b[idx_], road_idx).any() else road_idx
        # semantic = np.bincount(sem_b.squeeze()[idx_]).argmax() if not np.isin(sem_b[idx_], road_idx).any() else road_idx
        voxels[i] = hxyz[idx_][0]
        semantics[i] = semantic
        # points_f[i] = pcd_b[idx_].mean(axis=0) - center
        # points_f[i][2] += center[2] / 2
        # voxels.append(hxyz[idx_][0])
        # semantics.append(semantic)
        # points_f.append(pcd_b[idx_].mean(axis=0) - center)

    return voxels, semantics


def transform_pcd(xyz, transition):
    xyz[:, 1] *= -1
    xyz += np.asarray(transition)
    return xyz


def process_pcd(lidar_unprocessed, transition):
    xyz = lidar_unprocessed['data']['points_xyz']
    xyz = transform_pcd(xyz, transition)
    if lidar_unprocessed['data'].keys() == 2:
        intensity = lidar_unprocessed['data']['intensity']
        return np.concatenate([xyz, intensity[:, None]], axis=1)
    elif lidar_unprocessed['data'].keys() == 4:
        sem = lidar_unprocessed['data']['ObjTag']
        idx = lidar_unprocessed['data']['ObjIdx']
        cos = lidar_unprocessed['data']['CosAngel']
        return {'points': xyz, 'semantics': sem, 'ObjIdx': idx, 'CosAngel': cos}

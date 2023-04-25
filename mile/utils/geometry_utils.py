import numpy as np
import torch
from mile.data.dataset_utils import preprocess_gps


def bev_params_to_intrinsics(size, scale, offsetx):
    """
        size: number of pixels (width, height)
        scale: pixel size (in meters)
        offsetx: offset in x direction (direction of car travel)
    """
    intrinsics_bev = np.array([
        [1/scale, 0, size[0]/2 + offsetx],
        [0, -1/scale, size[1]/2],
        [0, 0, 1]
    ], dtype=np.float32)
    return intrinsics_bev


def intrinsics_inverse(intrinsics):
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    one = torch.ones_like(fx)
    zero = torch.zeros_like(fx)
    intrinsics_inv = torch.stack((
        torch.stack((1/fx, zero, -cx/fx), -1),
        torch.stack((zero, 1/fy, -cy/fy), -1),
        torch.stack((zero, zero, one), -1),
    ), -2)
    return intrinsics_inv


def get_out_of_view_mask(cfg):
    """ Returns a mask of everything that is not visible from the image given a certain bird's-eye view grid."""
    fov = cfg.IMAGE.FOV
    w = cfg.IMAGE.SIZE[1]
    resolution = cfg.BEV.RESOLUTION

    f = w / (2 * np.tan(fov * np.pi / 360.0))
    c_u = w / 2 - cfg.IMAGE.CROP[0]  # Adjust center point due to cropping

    bev_left = -np.round((cfg.BEV.SIZE[0] // 2) * resolution, decimals=1)
    bev_right = np.round((cfg.BEV.SIZE[0] // 2) * resolution, decimals=1)
    bev_bottom = 0.01
    # The camera is not exactly at the bottom of the bev image, so need to offset it.
    camera_offset = (cfg.BEV.SIZE[1] / 2 + cfg.BEV.OFFSET_FORWARD) * resolution + cfg.IMAGE.CAMERA_POSITION[0]
    bev_top = np.round(cfg.BEV.SIZE[1] * resolution - camera_offset, decimals=1)

    x, z = np.arange(bev_left, bev_right, resolution), np.arange(bev_bottom, bev_top, resolution)
    ucoords = x / z[:, None] * f + c_u

    # Return all points which lie within the camera bounds
    new_w = cfg.IMAGE.CROP[2] - cfg.IMAGE.CROP[0]
    mask = (ucoords >= 0) & (ucoords < new_w)
    mask = ~mask[::-1]
    mask_behind_ego_vehicle = np.ones((int(camera_offset / resolution), mask.shape[1]), dtype=np.bool)
    return np.vstack([mask, mask_behind_ego_vehicle])


def calculate_geometry(image_fov, height, width, forward, right, up, pitch, yaw, roll):
    """Intrinsics and extrinsics for a single camera.
    See https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/leaderboard/camera.py
    and https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/recording/sensors/camera.py
    """
    f = width / (2 * np.tan(image_fov * np.pi / 360.0))
    cx = width / 2
    cy = height / 2
    intrinsics = np.float32([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    extrinsics = get_extrinsics(forward, right, up, pitch, yaw, roll)
    return intrinsics, extrinsics


def get_extrinsics(forward, right, up, pitch, yaw, roll):
    # After multiplying the image coordinates by in the inverse intrinsics,
    # the resulting coordinates are defined with the axes (right, down, forward)
    assert pitch == yaw == roll == 0.0

    # After multiplying by the extrinsics, we want the axis to be (forward, left, up), and centered in the
    # inertial center of the ego-vehicle.
    mat = np.float32([
        [0,  0,  1, forward],
        [-1, 0,  0, -right],
        [0,  -1, 0, up],
        [0,  0,  0, 1],
    ])

    return mat


def lidar_to_histogram_features(lidar, cfg, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """

    # fit the center of the histogram the same as the bev.
    offset = cfg.BEV.OFFSET_FORWARD * cfg.BEV.RESOLUTION + cfg.IMAGE.CAMERA_POSITION[0]
    pixels_per_meter = cfg.POINTS.HISTOGRAM.RESOLUTION
    hist_max_per_pixel = cfg.POINTS.HISTOGRAM.HIST_MAX
    x_meters_max = cfg.POINTS.HISTOGRAM.X_MAX
    y_meters_max = cfg.POINTS.HISTOGRAM.Y_MAX

    def splat_points(point_cloud):
        # 256 x 256 grid
        xbins = np.linspace(
            -offset - x_meters_max,
            -offset + x_meters_max,
            2 * x_meters_max * pixels_per_meter + 1
        )
        ybins = np.linspace(
            -y_meters_max,
            y_meters_max,
            2 * y_meters_max * pixels_per_meter + 1,
        )
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist > hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist / hist_max_per_pixel
        return overhead_splat[::-1, ::-1]

    below = lidar[lidar[..., 2] <= 0]
    middle = lidar[(0 < lidar[..., 2]) & (lidar[..., 2] <= 2.5)]
    above = lidar[lidar[..., 2] > 2.5]
    below_features = splat_points(below)
    middle_features = splat_points(middle)
    above_features = splat_points(above)
    total_features = below_features + middle_features + above_features
    features = np.stack([below_features, middle_features, above_features, total_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features

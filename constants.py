import numpy as np


CARLA_FPS = 10
DISPLAY_SEGMENTATION = True
DISTORT_IMAGES = False
WHEEL_BASE = 2.8711279296875
# Ego-vehicle is 4.902m long and 2.128m wide. See `self._parent_actor.vehicle.bounding_box` in chaffeurnet_label
EGO_VEHICLE_DIMENSION = [4.902, 2.128, 1.511]

# https://github.com/carla-simulator/carla/blob/master/PythonAPI/carla/agents/navigation/local_planner.py
# However when processed, see "process_obs" function, unknown becomes lane_follow and the rest has a value between
# [0, 5] by substracting 1.
ROUTE_COMMANDS = {0: 'UNKNOWN',
                  1: 'LEFT',
                  2: 'RIGHT',
                  3: 'STRAIGHT',
                  4: 'LANEFOLLOW',
                  5: 'CHANGELANELEFT',
                  6: 'CHANGELANERIGHT',
                  }

BIRDVIEW_COLOURS = np.array([[255, 255, 255],          # Background
                             [225, 225, 225],       # Road
                             [160, 160, 160],      # Lane marking
                             [0, 83, 138],        # Vehicle
                             [127, 255, 212],      # Pedestrian
                             [50, 205, 50],        # Green light
                             [255, 215, 0],      # Yellow light
                             [220, 20, 60],        # Red light and stop sign
                             ], dtype=np.uint8)

# Obtained with sqrt of inverse frequency
SEMANTIC_SEG_WEIGHTS = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0])

VOXEL_LABEL = {
    0:   'Nothing',  # None
    1:   'Building',  # Building
    2:   'Fences',  # Fences
    3:   'Other',  # Other
    4:   'Pedestrian',  # Pedestrian
    5:   'Pole',  # Pole
    6:   'RoadLines',  # RoadLines
    7:   'Road',  # Road
    8:   'Sidewalk',  # Sidewalk
    9:   'Vegetation',  # Vegetation
    10:  'Vehicle',  # Vehicle
    11:  'Wall',  # Wall
    12:  'TrafficSign',  # TrafficSign
    13:  'Sky',  # Sky
    14:  'Ground',  # Ground
    15:  'Bridge',  # Bridge
    16:  'RailTrack',  # RailTrack
    17:  'GuardRail',  # GuardRail
    18:  'TrafficLight',  # TrafficLight
    19:  'Static',  # Static
    20:  'Dynamic',  # Dynamic
    21:  'Water',  # Water
    22:  'Terrain',  # Terrain
}


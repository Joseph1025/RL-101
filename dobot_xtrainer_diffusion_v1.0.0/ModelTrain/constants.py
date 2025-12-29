### Task parameters
import pathlib
import os

DATA_DIR = '/home/zz/User_dobot/datasets'
# DATA_DIR = '/home/zz/User_dobot/datasets/'
TASK_CONFIGS = {

    # dobot move cube new
    'demo_test1': {
        'dataset_dir': DATA_DIR + '/dataset_package_test/train_data',
        'episode_len': 1000,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'demo1_cube_cotrain': {
        'dataset_dir': [
            DATA_DIR + '/demo1_cube_data2/train_data',
            DATA_DIR + '/demo1_cube/train_data',
            DATA_DIR + '/dataset_package_new3/train_data',
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/demo1_cube_data2/train_data',
        ],
        'sample_weights': [4, 3,3],
        'train_ratio': 0.9,  # ratio of train data from the first dataset_dir
        'episode_len': 1000,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

}

###  fixed constants
DT = 0.02
FPS = 50





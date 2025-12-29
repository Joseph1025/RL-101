import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
robomimic_path = os.path.join(BASE_DIR, 'robomimic')
print(robomimic_path)
sys.path.append(robomimic_path)
from ModelTrain.module.model_module import Imitate_Model
import cv2

import h5py
import numpy as np
import cv2


if __name__ == '__main__':
    model_name = 'policy_last.ckpt'
    model = Imitate_Model(ckpt_dir='./ckpt/test', ckpt_name=model_name)
    model.loadModel()
    observation = {'qpos':[],'images':{'left_wrist':[],'right_wrist':[],'top':[]}}
    last_action=[1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    show_canvas = np.zeros((480, 640*3, 3), dtype=np.uint8)
    # pick a hdf5 file as the real-action and rgb data to test the ouput of the model
    with h5py.File("/home/zz/User_dobot/datasets/demo_cube/train_data/episode_init_5.hdf5", 'r', rdcc_nbytes=1024 ** 2 * 2) as root:
        print(len(root["/observations/images/top"]))
        for i in range(len(root["/observations/images/top"])):
            qpos = root["/observations/qpos"][i]
            print("qpos:",[np.rad2deg(i) for i in qpos])
            action = root["action"][i]
            print("action:",[np.rad2deg(i) for i in action])
            # press enter to continue
            show_canvas[:, :640] = np.asarray(cv2.imdecode(np.asarray(root["/observations/images/top"][i], dtype="uint8"), cv2.IMREAD_COLOR), dtype="uint8")
            show_canvas[:, 640:640 * 2] = np.asarray(cv2.imdecode(np.asarray(root["/observations/images/left_wrist"][i], dtype="uint8"), cv2.IMREAD_COLOR), dtype="uint8")
            show_canvas[:, 640 * 2:640 * 3] = np.asarray(cv2.imdecode(np.asarray(root["/observations/images/right_wrist"][i], dtype="uint8"), cv2.IMREAD_COLOR), dtype="uint8")
            cv2.imshow("0", show_canvas)

            observation['qpos'] = qpos  #  input joint value (unit radians) and Grippers value(0~1).The 7th and 14th values are the left and right hand gripper values, respectively
            observation['images']['left_wrist'] = cv2.imdecode(np.asarray(root["/observations/images/left_wrist"][i], dtype="uint8"), cv2.IMREAD_COLOR)  # input image
            observation['images']['right_wrist'] = cv2.imdecode(np.asarray(root["/observations/images/right_wrist"][i], dtype="uint8"), cv2.IMREAD_COLOR)
            observation['images']['top'] = cv2.imdecode(np.asarray(root["/observations/images/top"][i], dtype="uint8"), cv2.IMREAD_COLOR)
            print('left_wrise', observation['images']['left_wrist'].shape)
            print('right_wrise', observation['images']['right_wrist'].shape)     
            print('top', observation['images']['top'].shape)       
            predict_action = model.predict(observation, i)  # output
            print("action_delta:",[np.rad2deg(i) for i in (predict_action-action)])
            print("action_increasement:",[np.rad2deg(i) for i in (predict_action-last_action)])
            last_action=predict_action

            cv2.waitKey(0)

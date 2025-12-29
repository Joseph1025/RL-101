import numpy as np
import torch
import os
import h5py
# import pickle
import fnmatch
import cv2
# from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

# import PIL.Image as Image
# import matplotlib.pyplot as plt

import IPython
e = IPython.embed

def flatten_list(l):
    return [item for sublist in l for item in sublist]

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class,use_depth_image=False):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        # if self.policy_class == 'Diffusion':  #-src code #yzhcomment
        #     self.augment_images = True
        # else:
        # self.augment_images = False
        self.augment_images = True # change code
        self.transformations = None
        self.__getitem__(0) # initialize self.is_sim and self.transformations
        self.is_sim = False
        self.use_depth_image = use_depth_image


    def __len__(self):
        return len(self.episode_ids)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        try:
            # print(dataset_path)
            with h5py.File(dataset_path, 'r') as root:
                try: # some legacy data does not have this attribute
                    is_sim = root.attrs['sim']
                except:
                    is_sim = False
                compressed = root.attrs.get('compress', False)
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:  
                    action = root['/action'][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
                original_action_shape = action.shape
                episode_len = original_action_shape[0]
                # # get observation at start_ts only
                # qpos = root['/observations/qpos'][start_ts]
                # qvel = root['/observations/qvel'][start_ts]
                # image_dict = dict()
                # depth_dict = dict()
                # image_depth_dict = dict()
                # for cam_name in self.camera_names:
                #     image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

                ############################## ADDED THE STUFF BELOW #####################################
                # Get observation history based on observation_horizon
                observation_horizon = 2  # Or read from self somewhere
                qpos_list = []
                image_dict_list = []

                for t_offset in range(observation_horizon):
                    t = max(0, start_ts - observation_horizon + 1 + t_offset)  # Handle boundary
                    qpos_list.append(root['/observations/qpos'][t])
                    
                    temp_image_dict = dict()
                    for cam_name in self.camera_names:
                        temp_image_dict[cam_name] = root[f'/observations/images/{cam_name}'][t]
                    image_dict_list.append(temp_image_dict)

                # Stack qpos history
                qpos = np.stack(qpos_list, axis=0)  # Shape: (observation_horizon, 14)

                # Stack image history  
                image_dict = dict()
                for cam_name in self.camera_names:
                    images = [img_dict[cam_name] for img_dict in image_dict_list]
                    image_dict[cam_name] = np.stack(images, axis=0)  # Shape: (observation_horizon, H, W, C)
                # ADDED THE STUFF ABOVE
                ##########################################################################################
                
                # if compressed:
                #     for cam_name in image_dict.keys():
                #         decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                #         image_dict[cam_name] = np.array(decompressed_image)

                ##########################################################################################
                if compressed:
                    for cam_name in image_dict.keys():
                        # image_dict[cam_name] is (observation_horizon, compressed_bytes)
                        decompressed_frames = []
                        for t in range(len(image_dict[cam_name])):
                            decompressed = cv2.imdecode(image_dict[cam_name][t], 1)
                            decompressed_frames.append(np.array(decompressed))
                        image_dict[cam_name] = np.stack(decompressed_frames, axis=0)
                ##########################################################################################

                # get all actions after and including start_ts
                if is_sim:
                    action = action[start_ts:]
                    action_len = episode_len - start_ts
                else:
                    action = action[max(0, start_ts - 1):] # hack, to make timesteps more aligned
                    action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

            # self.is_sim = is_sim
            padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
            padded_action[:action_len] = action
            is_pad = np.zeros(self.max_episode_len)
            is_pad[action_len:] = 1

            padded_action = padded_action[:self.chunk_size]
            is_pad = is_pad[:self.chunk_size]

            # new axis for different cameras
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=1)  # (observation_horizon, num_cameras, H, W, C)

            # # construct observations
            # image_data = torch.from_numpy(all_cam_images)
            # qpos_data = torch.from_numpy(qpos).float()

            ##########################################################################################
            # Change to handle temporal dimension:
            # all_cam_images shape: (observation_horizon, num_cameras, H, W, C)
            # Need to reshape for proper batching
            image_data = torch.from_numpy(all_cam_images)  
            qpos_data = torch.from_numpy(qpos).float()
            # Shapes will be: image_data: (observation_horizon, num_cameras, H, W, C)
            #                 qpos_data: (observation_horizon, 14)
            ##########################################################################################

            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            # channel last
            # image_data = torch.einsum('k h w c -> k c h w', image_data)
            # channel last - handle temporal dimension
            # Shape: (observation_horizon, num_cameras, H, W, C) -> (observation_horizon, num_cameras, C, H, W)
            image_data = torch.einsum('t k h w c -> t k c h w', image_data)
            image_data = image_data / 255.0

            # # Apply random crop/resize augmentation during training
            # if self.augment_images:
            #     # image_data shape: (num_cameras, C, H, W)
            #     num_cams, C, H, W = image_data.shape
            #     ratio = 0.95
                
            #     # Calculate crop size
            #     crop_h = int(H * ratio)
            #     crop_w = int(W * ratio)
                
            #     # RANDOM crop (not center) - this is key for augmentation!
            #     import random
            #     top = random.randint(0, H - crop_h)
            #     left = random.randint(0, W - crop_w)
                
            #     # Apply same crop to all cameras for consistency
            #     image_data = image_data[:, :, top:top+crop_h, left:left+crop_w]
                
            #     # Resize back to original size
            #     resize_transform = transforms.Resize((H, W), antialias=True)
            #     image_data = resize_transform(image_data)

            # Apply random crop/resize augmentation during training
            
            if self.augment_images:
                # image_data shape: (observation_horizon, num_cameras, C, H, W)
                T, num_cams, C, H, W = image_data.shape
                ratio = 0.95
                
                # Calculate crop size
                crop_h = int(H * ratio)
                crop_w = int(W * ratio)
                
                # RANDOM crop (not center) - this is key for augmentation!
                import random
                top = random.randint(0, H - crop_h)
                left = random.randint(0, W - crop_w)
                
                # Apply same crop to all temporal frames and cameras for consistency
                image_data = image_data[:, :, :, top:top+crop_h, left:left+crop_w]
                
                # Resize back to original size
                resize_transform = transforms.Resize((H, W), antialias=True)
                # Need to reshape to apply resize: (T, num_cams, C, crop_h, crop_w)
                # Resize expects (N, C, H, W), so we'll process each temporal frame
                resized_frames = []
                for t in range(T):
                    resized_cams = []
                    for cam in range(num_cams):
                        resized_cam = resize_transform(image_data[t, cam])  # (C, H, W)
                        resized_cams.append(resized_cam)
                    resized_frames.append(torch.stack(resized_cams, dim=0))
                image_data = torch.stack(resized_frames, dim=0)  # (T, num_cams, C, H, W)
            
            # Transpose to (num_cams, T, C, H, W) for policy network
            # This must happen whether augmentation is applied or not
            image_data = image_data.transpose(0, 1)  # Swap T and num_cams dimensions

            # src code
            if self.policy_class == 'Diffusion':
                # normalize to [-1, 1]
                action_data = ((action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
            else:
                # normalize to mean 0 std 1
                action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]


        except:
            print(f'Error loading {dataset_path} in __getitem__')
            quit()

        # print(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)
        # print(image_data.device, qpos_data.device, action_data.device, is_pad.device)
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_path_list):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:
                    action = root['/action'][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": qpos}

    return stats, all_episode_len

def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, skip_mirrored_data=False, load_pretrain=False, policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.99):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # obtain train test split on dataset_dir_l[0]
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')

    _, all_episode_len = get_norm_stats(dataset_path_list)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]))
    print(f'Norm stats from: {stats_dir_l}')

    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    # print(f'train_episode_len: {train_episode_len}, val_episode_len: {val_episode_len}, train_episode_ids: {train_episode_ids}, val_episode_ids: {val_episode_ids}')

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, policy_class)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, policy_class)
    # Use 16 workers for augmented data, 2 for non-augmented
    train_num_workers = 16 if train_dataset.augment_images else 2
    val_num_workers = 8 if train_dataset.augment_images else 2
    # train_num_workers =  1  # YL change
    # val_num_workers =  1  # YL change
    print(f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')

    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=train_num_workers, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=val_num_workers, prefetch_factor=2)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0 # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action

def smooth_base_action(base_action):
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5)/5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)

def preprocess_base_action(base_action):
    # base_action = calibrate_linear_vel(base_action)
    base_action = smooth_base_action(base_action)

    return base_action

def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0
    # angular_vel = 0
    # if np.abs(linear_vel) < 0.05:
    #     linear_vel = 0
    return np.array([linear_vel, angular_vel])

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

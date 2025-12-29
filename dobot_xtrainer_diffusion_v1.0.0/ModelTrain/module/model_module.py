# 导入必要的库和模块  
# Import necessary libraries and modules
import torch
import numpy as np
import os
import pickle
from einops import rearrange  # 用于重新排列张量的维度 / For rearranging tensor dimensions
import matplotlib.pyplot as plt
import time
from torchvision import transforms

# 导入自定义模块 / Import custom modules
from ModelTrain.module.policy import DiffusionPolicy  # 扩散策略模型 / Diffusion strategy model
from detr.models.latent_model import Latent_Model_Transformer  # 潜在模型Transformer / Latent Model Transformer
from ModelTrain.model_train import arg_config  # 参数配置函数 / Argument configuration function

# 定义配置函数 / Define configuration function
def set_config():
    args = arg_config()  # 获取参数配置 / Retrieve parameter configurations
    args["chunk_size"] = 48  # 设置块大小为48 / Set chunk size to 48
    ckpt_dir = args["ckpt_dir"]  # 检查点目录 / Checkpoint directory
    policy_class = "Diffusion"  # 策略类别为“扩散” / Policy class set to "Diffusion"
    task_name = args["task_name"]  # 任务名称 / Task name
    batch_size_train = args["batch_size"]  # 训练批次大小 / Training batch size
    batch_size_val = args["batch_size"]  # 验证批次大小 / Validation batch size
    num_steps = args["num_steps"]  # 总训练步骤数 / Total number of training steps
    eval_every = args["eval_every"]  # 每隔多少步评估一次 / Evaluate every N steps
    validate_every = args["validate_every"]  # 每隔多少步验证一次 / Validate every N steps
    save_every = args["save_every"]  # 每隔多少步保存一次模型 / Save the model every N steps
    resume_ckpt_path = args["resume_ckpt_path"]  # 继续训练的检查点路径 / Path to resume training checkpoint
    is_sim = task_name[:4] == "sim_"  # 判断是否为模拟任务 / Check if the task is a simulation task

    # 根据任务名称加载对应的任务配置 / Load task configuration based on the task name
    if is_sim or task_name == "all":
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]

    # 从任务配置中提取参数 / Extract parameters from task configuration
    dataset_dir = task_config["dataset_dir"]  # 数据集目录 / Dataset directory
    episode_len = task_config["episode_len"]  # 每个episode的长度 / Length of each episode
    camera_names = task_config["camera_names"]  # 摄像头名称列表 / List of camera names
    stats_dir = task_config.get("stats_dir", None)  # 统计数据目录 / Statistics directory
    sample_weights = task_config.get("sample_weights", None)  # 样本权重 / Sample weights
    train_ratio = task_config.get("train_ratio", 0.99)  # 训练集比例 / Training set ratio
    name_filter = task_config.get("name_filter", lambda n: True)  # 名称过滤器 / Name filter
    state_dim = 14  # 状态维度 / State dimension
    lr_backbone = 1e-5  # 主干网络的学习率 / Learning rate of the backbone network
    backbone = "resnet18"  # 主干网络类型 / Type of backbone network

    # 根据策略类别设置策略配置 / Configure policy settings based on policy class
    if policy_class == "Diffusion":
        policy_config = {
            'lr': args["lr"],  # 学习率 / Learning rate
            'camera_names': camera_names,  # 摄像头名称 / Camera names
            'action_dim': 16,  # 动作维度 / Action dimension
            'observation_horizon': 2, # CHAN ADDED THIS!
            # 'observation_horizon': 1,  # 观测范围 / Observation horizon
            'action_horizon': 8,  # 动作范围 / Action horizon
            # 'prediction_horizon': args["chunk_size"],  # 预测范围 / Prediction horizon
            # 'num_queries': args["chunk_size"],  # 查询次数 / Number of queries
            'prediction_horizon': 16,   # Predict 16 steps ahead
            'num_queries': 16,          # Re-query every 16 steps  
            'num_inference_timesteps': 10,  # 推断时间步数 / Number of inference timesteps
            'ema_power': 0.75,  # EMA系数 / EMA coefficient
            'vq': False  # 是否使用向量量化 / Whether to use vector quantization
        }
    else:
        raise NotImplementedError

    # 汇总所有配置 / Combine all configurations
    config = {
        'num_steps': num_steps,  # 总训练步骤数 / Total training steps
        'eval_every': eval_every,  # 评估间隔 / Evaluation interval
        'validate_every': validate_every,  # 验证间隔 / Validation interval
        'save_every': save_every,  # 保存间隔 / Save interval
        'ckpt_dir': ckpt_dir,  # 检查点目录 / Checkpoint directory
        'resume_ckpt_path': resume_ckpt_path,  # 继续训练路径 / Resume training path
        'episode_len': episode_len,  # 每个episode的长度 / Episode length
        'state_dim': state_dim,  # 状态维度 / State dimension
        'lr': args["lr"],  # 学习率 / Learning rate
        'policy_class': policy_class,  # 策略类别 / Policy class
        'policy_config': policy_config,  # 策略配置 / Policy configuration
        'task_name': task_name,  # 任务名称 / Task name
        'seed': args["seed"],  # 随机种子 / Random seed
        'temporal_agg': args["temporal_agg"],  # 时间聚合 / Temporal aggregation
        'camera_names': camera_names,  # 摄像头名称 / Camera names
        'real_robot': not is_sim,  # 是否是真实机器人 / Whether it is a real robot
        'load_pretrain': args["load_pretrain"]  # 是否加载预训练模型 / Whether to load pre-trained model
    }
    return config


class Imitate_Model:

    def __init__(self, ckpt_dir=None, ckpt_name='policy_last.ckpt'):
        config = set_config()  # 获取配置 / Get configuration
        self.ckpt_name = ckpt_name  # 检查点文件名 / Checkpoint file name

        # 设置检查点目录 / Set checkpoint directory
        if ckpt_dir is None:
            self.ckpt_dir = config["ckpt_dir"]
            print(self.ckpt_dir)
        else:
            self.ckpt_dir = ckpt_dir

        # 从配置中获取参数 / Get parameters from configuration
        self.state_dim = config["state_dim"]
        self.policy_class = config["policy_class"]
        self.policy_config = config["policy_config"]
        self.camera_names = config["camera_names"]
        self.max_timesteps = config["episode_len"]
        self.temporal_agg = config["temporal_agg"]
        self.vq = config["policy_config"]["vq"]
        self.t = 0  # 时间步计数 / Time step counter
        self.image_history = []

    # 私有方法，创建策略模型
    def __make_policy(self):
        if self.policy_class == "Diffusion":
            policy = DiffusionPolicy(self.policy_config)  # 创建扩散策略模型
        else:
            raise NotImplementedError
        return policy

    # 私有方法，处理图像数据  
    # Private method to process image data
    def __image_process(self, observation, camera_names, rand_crop_resize=False):
        curr_images = []
        for cam_name in camera_names:
            # 获取摄像头图像并调整维度 / Get camera image and adjust dimensions
            curr_image = rearrange(observation["images"][cam_name], "h w c -> c h w")
            curr_images.append(curr_image)

        # 将所有图像堆叠并转换为Tensor / Stack all images and convert to Tensor
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

        rand_crop_resize = False  # 默认不进行随机裁剪和调整大小 / Default to no random crop and resize
        if rand_crop_resize:
            print("rand crop resize is used!")
            original_size = curr_image.shape[-2]  # 原始尺寸 / Original size
            ratio = 0.95  # 裁剪比例 / Crop ratio
            height = curr_image.shape[-2]
            width = curr_image.shape[-1]

            # 计算裁剪区域 / Calculate crop area
            height_start = int((1 - ratio) * height / 2)
            height_end = height - height_start
            width_start = int((1 - ratio) * width / 2)
            width_end = width - width_start

            # 裁剪图像 / Crop the image
            curr_image = curr_image[:, :, height_start:height_end, width_start:width_end]
            curr_image = curr_image.squeeze(0)

            # 调整回原始尺寸 / Resize to original size
            resize_transform = transforms.Resize(original_size, antialias=True)
            curr_image = resize_transform(curr_image)
            curr_image = curr_image.unsqueeze(0)
        return curr_image

    ########################### ADDED THIS METHOD!
    def __image_process_history(self, stacked_images_dict, camera_names):
        """
        Process temporal stack of images for observation_horizon > 1
        stacked_images_dict: Dict with keys=camera names, values=np.array(T, H, W, C)
        Returns: torch.tensor of shape (1, num_cameras, T, C, H, W) or properly shaped for model
        """
        observation_horizon = 2
        curr_images = []
        
        for cam_name in camera_names:
            # stacked_images_dict[cam_name] shape: (T, H, W, C) where T = observation_horizon
            cam_images_temporal = stacked_images_dict[cam_name]
            
            # Process each temporal frame
            temporal_stack = []
            for t in range(len(cam_images_temporal)):
                # Rearrange each frame: (H, W, C) -> (C, H, W)
                frame = rearrange(cam_images_temporal[t], "h w c -> c h w")
                temporal_stack.append(frame)
            
            # Stack temporal frames: (T, C, H, W)
            cam_temporal = np.stack(temporal_stack, axis=0)
            curr_images.append(cam_temporal)
        
        # Stack all cameras: (num_cameras, T, C, H, W)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        # Result shape: (1, num_cameras, T, C, H, W)
        
        # NO AUGMENTATION during inference
        rand_crop_resize = False
        
        return curr_image

    # 私有方法，自动获取日志索引  
    # Private method to automatically get log index
    def __get_auto_index(self, dataset_dir):
        max_idx = 1000  # 最大索引 / Maximum index
        for i in range(max_idx + 1):
            # 如果文件不存在，返回当前索引 / If the file doesn't exist, return the current index
            if not os.path.isfile(os.path.join(dataset_dir, f"qpos_{i}.npy")):
                return i
        else:
            # 超过最大索引，抛出异常 / Raise an exception if exceeding maximum index
            raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

    # 加载模型  
    # Load the model
    def loadModel(self):
        # 获取当前文件和目录路径 / Get the current file and directory path
        cur_path = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.dirname(cur_path)
        print(dir_path)

        # 构建检查点路径 / Construct the checkpoint path
        ckpt_path = os.path.join(self.ckpt_dir, self.ckpt_name)
        print(ckpt_path)
        ckpt_path = dir_path + ckpt_path[1:]  # 调整路径格式 / Adjust path format
        print(ckpt_path)

        # 创建策略模型并加载权重 / Create the policy model and load weights
        self.policy = self.__make_policy()
        loading_status = self.policy.deserialize(torch.load(ckpt_path))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()

        # 如果使用向量量化，加载潜在模型 / Load latent model if vector quantization is used
        if self.vq:
            vq_dim = self.config["policy_config"]["vq_dim"]
            vq_class = self.config["policy_config"]["vq_class"]
            latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
            latent_model_ckpt_path = os.path.join(self.ckpt_dir, "latent_model_last.ckpt")
            latent_model.deserialize(torch.load(latent_model_ckpt_path))
            latent_model.eval()
            latent_model.cuda()
            print(f"Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}")
        else:
            print(f"Loaded: {ckpt_path}")


        # 加载统计数据 / Load statistics data
        stats_path = os.path.join(dir_path, self.ckpt_dir, "dataset_stats.pkl")
        print(f"Loaded stats from: {stats_path}")
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        # 定义预处理和后处理函数 / Define preprocessing and postprocessing functions
        self.pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
        if self.policy_class == "Diffusion":
            self.post_process = lambda a: (a + 1) / 2 * (stats["action_max"] - stats["action_min"]) + stats["action_min"]
        else:
            self.post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

        # 设置查询频率 / Set query frequency
        self.query_frequency = self.policy_config["num_queries"]
        if self.temporal_agg:
            self.query_frequency = 1
            self.num_queries = self.policy_config["num_queries"]
        self.max_timesteps = int(self.max_timesteps * 1)

        # 初始化变量 / Initialize variables
        self.episode_returns = []
        self.highest_rewards = []
        if self.temporal_agg:
            self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps + self.num_queries, 16]).cuda()
        self.qpos_history_raw = np.zeros((self.max_timesteps, self.state_dim))
        self.image_list = []
        self.qpos_list = []
        self.target_qpos_list = []
        self.rewards = []
        self.all_actions = []

    # 预测函数 / Prediction Function
    def predict(self, observation, t, save_qpos_history=False):
        with torch.inference_mode():  # 启用推理模式 / Enable inference mode
            # 获取观测的qpos并记录 / Get and store the observed qpos
            qpos_numpy = np.array(observation["qpos"])
            self.qpos_history_raw[t] = qpos_numpy

            # # 对qpos进行预处理 / Preprocess the qpos
            # qpos = self.pre_process(qpos_numpy)
            # qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

            ############################ Added below ##########################################
            # Build observation history
            observation_horizon = 2
            qpos_history = []
            for i in range(observation_horizon):
                t_hist = max(0, t - observation_horizon + 1 + i)
                qpos_history.append(self.qpos_history_raw[t_hist])

            # Stack and preprocess
            qpos_stacked = np.stack(qpos_history, axis=0)  # Shape: (2, 14)
            qpos = self.pre_process(qpos_stacked)  # Normalize each frame
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)  # Shape: (1, 2, 14)
            #################################### ADDED ABOVE #########################################

            # # 每隔query_frequency步处理一次图像 / Process the image every query_frequency steps
            # if t % self.query_frequency == 0:
            #     curr_image = self.__image_process(
            #         observation,
            #         self.camera_names,
            #         rand_crop_resize=(self.policy_class == "Diffusion")
            #     )

            ################################# ADDED BELOW #####################################
            # Change to: maintain image history buffer
            if not hasattr(self, 'image_history'):
                self.image_history = []

            # Add current images to history
            current_images = {}
            for cam_name in self.camera_names:
                current_images[cam_name] = observation["images"][cam_name]
            self.image_history.append(current_images)

            # Keep only last observation_horizon frames
            if len(self.image_history) > observation_horizon:
                self.image_history = self.image_history[-observation_horizon:]

            # Process history
            if t % self.query_frequency == 0:
                # Stack frames from history
                stacked_images = {}
                for cam_name in self.camera_names:
                    frames = [self.image_history[i][cam_name] for i in range(len(self.image_history))]
                    stacked_images[cam_name] = np.stack(frames, axis=0)  # (T, H, W, C)
                
                curr_image = self.__image_process_history(stacked_images, self.camera_names)
            ####################################################################################

            if t == 0:  # 网络预热，运行多次前向传播 / Network warm-up with multiple forward passes
                for _ in range(10):
                    self.policy(qpos, curr_image)
                else:
                    print("network warm up done")
                    time1 = time.time()

            if self.policy_class == "Diffusion":  # 判断策略类别为“Diffusion” / Check if policy class is "Diffusion"
                if t % self.query_frequency == 0:
                    # 获取所有预测的动作 / Get all predicted actions
                    print("qpos:", qpos.shape)
                    print("curr_image:", curr_image.shape)
                    self.all_actions = self.policy(qpos, curr_image).detach()
                if self.temporal_agg:  # 时间聚合，平滑动作序列 / Temporal aggregation for action smoothing
                    self.all_time_actions[[t], t:t + self.num_queries] = self.all_actions
                    actions_for_curr_step = self.all_time_actions[:, t]
                    actions_populated = torch.all((actions_for_curr_step != 0), axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.014  # 衰减系数 / Decay factor
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    # 获取当前时间步的动作 / Get the action for the current time step
                    raw_action = self.all_actions[:, t % self.query_frequency]

            # 后处理动作 / Post-process the action
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = self.post_process(raw_action)
            target_qpos = action[:-2]  # 目标qpos / Target qpos
            base_action = action[-2:]  # 基础动作 / Base action

            # 记录qpos和目标qpos / Store qpos and target qpos
            self.qpos_list.append(qpos_numpy)
            self.target_qpos_list.append(target_qpos)

            if save_qpos_history:  # 如果需要，保存qpos历史记录 / Save qpos history if required
                log_id = self.__get_auto_index(self.ckpt_dir)
                np.save(os.path.join(self.ckpt_dir, f"qpos_{log_id}.npy"), self.qpos_history_raw)

                # 绘制并保存qpos曲线 / Plot and save the qpos curve
                plt.figure(figsize=(10, 20))
                for i in range(self.state_dim):
                    plt.subplot(self.state_dim, 1, i + 1)
                    plt.plot(self.qpos_history_raw[:, i])
                    if i != self.state_dim - 1:
                        plt.xticks([])  # 隐藏x轴刻度 / Hide x-axis ticks
                plt.tight_layout()
                plt.savefig(os.path.join(self.ckpt_dir, f"qpos_{log_id}.png"))
                plt.close()

        return target_qpos  # 返回目标qpos / Return the target qpos

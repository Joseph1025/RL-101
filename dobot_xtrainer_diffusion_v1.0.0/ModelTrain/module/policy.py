import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
import IPython
e = IPython.embed
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
robomimic_path = os.path.join(BASE_DIR, 'robomimic')
sys.path.append(robomimic_path)
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D
print(" correctly input robomimic")
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel


# 定义一个名为 DiffusionPolicy 的神经网络模型，继承自 nn.Module
# Define a neural network model named DiffusionPolicy, inheriting from nn.Module
class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()

        # 从参数中获取摄像头名称列表
        # Get the list of camera names from the parameters
        self.camera_names = args_override['camera_names']

        # 定义观测、动作和预测的时间步长
        # Define the time horizons for observation, action, and prediction
        self.observation_horizon = args_override['observation_horizon']  # 观测序列长度 / Observation sequence length
        self.action_horizon = args_override['action_horizon']  # 动作序列长度 / Action sequence length
        self.prediction_horizon = args_override['prediction_horizon']  # 预测序列长度 / Prediction sequence length

        # 定义推理时的时间步数和 EMA 衰减率
        # Define the number of inference timesteps and EMA decay rate
        self.num_inference_timesteps = args_override['num_inference_timesteps']
        self.ema_power = args_override['ema_power']

        # 学习率和权重衰减
        # Learning rate and weight decay
        self.lr = args_override['lr']
        self.weight_decay = 0  # 默认为 0 / Default is 0

        # 定义关键点数量和特征维度
        # Define the number of keypoints and feature dimensions
        self.num_kp = 32  # 关键点数量 / Number of keypoints
        self.feature_dimension = 64  # 每个摄像头的特征维度 / Feature dimension for each camera

        # 动作和观测的维度
        # Define the dimensions for action and observation
        self.ac_dim = args_override['action_dim']  # 动作维度，例如 14 + 2 / Action dimension, e.g., 14 + 2
        self.obs_dim = self.feature_dimension * len(self.camera_names) + 14  # 观测维度，包括摄像头特征和机器人状态 / Observation dimension, including camera features and robot states

        # 定义用于视觉特征提取的网络模块
        # Define the network modules for visual feature extraction
        backbones = []
        pools = []
        linears = []
        for _ in self.camera_names:
            # 定义卷积神经网络（ResNet18），不使用预训练权重
            # Define a convolutional neural network (ResNet18) without pretrained weights
            backbones.append(ResNet18Conv(**{
                'input_channel': 3,
                'pretrained': False,
                'input_coord_conv': False
            }))

            # 使用 SpatialSoftmax 提取空间特征
            # Use SpatialSoftmax to extract spatial features
            pools.append(SpatialSoftmax(**{
                'input_shape': [512, 15, 20],  # 输入特征图的形状 / Shape of the input feature map
                'num_kp': self.num_kp,  # 关键点数量 / Number of keypoints
                'temperature': 1.0,
                'learnable_temperature': False,
                'noise_std': 0.0
            }))

            # 将关键点展平后通过线性层降维
            # Flatten the keypoints and reduce dimensions using a linear layer
            linears.append(torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))

        # 将列表转换为 ModuleList，以便于模型的管理和参数注册
        # Convert the lists to ModuleList for model management and parameter registration
        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)

        # 将批归一化层替换为组归一化层，可能是为了更好的泛化性能
        # Replace batch normalization with group normalization for better generalization
        backbones = replace_bn_with_gn(backbones)

        # 定义降噪网络，使用条件 Unet1D
        # Define the noise prediction network using a conditional Unet1D
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim * self.observation_horizon  # 全局条件的维度 / Dimension of the global condition
        )

        # 将所有网络模块放入一个字典中，方便管理
        # Store all network modules in a dictionary for easier management
        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbones': backbones,  # 特征提取网络 / Feature extraction networks
                'pools': pools,  # 空间软最大值池化 / Spatial softmax pooling
                'linears': linears,  # 线性层 / Linear layers
                'noise_pred_net': noise_pred_net  # 降噪网络 / Noise prediction network
            })
        })

        # 将模型转换为浮点类型并移动到 GPU
        # Convert the model to float and move it to GPU
        nets = nets.float().cuda()

        # 是否启用 EMA（指数移动平均）模型
        # Whether to enable EMA (Exponential Moving Average) model
        ENABLE_EMA = True
        if ENABLE_EMA:
            ema = EMAModel(model=nets, power=self.ema_power)
        else:
            ema = None

        self.nets = nets  # 网络模型 / Network model
        self.ema = ema  # EMA 模型 / EMA model

        # 设置噪声调度器，使用 DDIM 调度器
        # Set the noise scheduler using a DDIM scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,  # 训练时的时间步数 / Number of timesteps during training
            beta_schedule='squaredcos_cap_v2',  # beta 调度策略 / Beta scheduling strategy
            clip_sample=True,  # 是否裁剪样本 / Whether to clip samples
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'  # 预测类型 / Prediction type
        )

        # 打印模型的参数数量
        # Print the number of parameters in the model
        n_parameters = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    # 配置优化器
    # Configure the optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    # 定义前向传播函数
    # Define the forward pass function
    def __call__(self, qpos, image, actions=None, is_pad=None):
        B = qpos.shape[0]  # 获取批次大小 / Get the batch size

        if actions is not None:  # 训练阶段 / Training phase
            nets = self.nets
            all_features = []
            # for cam_id in range(len(self.camera_names)):
            #     cam_image = image[:, cam_id]  # 获取第 cam_id 个摄像头的图像 / Get the image from camera cam_id
            #     cam_features = nets['policy']['backbones'][cam_id](cam_image)  # 提取卷积特征 / Extract convolutional features
            #     pool_features = nets['policy']['pools'][cam_id](cam_features)  # 空间软最大值池化 / Spatial softmax pooling
            #     pool_features = torch.flatten(pool_features, start_dim=1)  # 展平特征 / Flatten the features
            #     out_features = nets['policy']['linears'][cam_id](pool_features)  # 线性变换 / Linear transformation
            #     all_features.append(out_features)

            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]  # (B, T, C, H, W) for observation_horizon=T
                
                # Process each temporal frame separately
                temporal_features = []
                for t in range(cam_image.shape[1]):  # Loop over observation_horizon
                    frame = cam_image[:, t]  # (B, C, H, W)
                    cam_features = nets['policy']['backbones'][cam_id](frame)
                    pool_features = nets['policy']['pools'][cam_id](cam_features)
                    pool_features = torch.flatten(pool_features, start_dim=1)
                    out_features = nets['policy']['linears'][cam_id](pool_features)  # (B, feature_dim)
                    temporal_features.append(out_features)
                
                # Stack temporal features: (B, T, feature_dim)
                temporal_features = torch.stack(temporal_features, dim=1)
                all_features.append(temporal_features)            

            # # 拼接所有摄像头的特征和机器人状态作为观测条件
            # # Concatenate all camera features and robot states as observation conditions
            # obs_cond = torch.cat(all_features + [qpos], dim=1)

            ############################ ADDED BELOW ############################
            # qpos shape: (B, observation_horizon, 14)
            # all_features: list of (B, observation_horizon, feature_dim)
            # Need to flatten temporal dimension
            qpos_flat = qpos.reshape(B, -1)  # (B, observation_horizon * 14)
            features_flat = [f.reshape(B, -1) for f in all_features]  # Flatten each
            obs_cond = torch.cat(features_flat + [qpos_flat], dim=1)
            # Result: (B, obs_dim * observation_horizon)
            ######################################################################

            # 为动作添加噪声
            # Add noise to the actions
            noise = torch.randn(actions.shape, device=obs_cond.device)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=obs_cond.device
            ).long()

            # 生成带有噪声的动作 / Generate noisy actions
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

            # 预测噪声 / Predict the noise
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

            # 计算损失（均方误差）/ Calculate the loss (MSE)
            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()  # 考虑填充的情况 / Handle padding

            # 构建损失字典 / Create a loss dictionary
            loss_dict = {'l2_loss': loss, 'loss': loss}

            # 更新 EMA 模型 / Update the EMA model
            if self.training and self.ema is not None:
                self.ema.step(nets)

            return loss_dict

        else:  # 推理阶段 / Inference phase
            To = self.observation_horizon
            Ta = self.action_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim
            print('action_dim:', action_dim)

            nets = self.nets
            if self.ema is not None:
                nets = self.ema.averaged_model  # 使用 EMA 模型 / Use the EMA model

            all_features = []
            # for cam_id in range(len(self.camera_names)):
            #     cam_image = image[:, cam_id]
            #     cam_features = nets['policy']['backbones'][cam_id](cam_image)
            #     pool_features = nets['policy']['pools'][cam_id](cam_features)
            #     pool_features = torch.flatten(pool_features, start_dim=1)
            #     out_features = nets['policy']['linears'][cam_id](pool_features)
            #     all_features.append(out_features)

            ########################### ADDED BELOW ######################################################
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]  # (B, T, C, H, W) for observation_horizon=T
                
                # Process each temporal frame separately
                temporal_features = []
                for t in range(cam_image.shape[1]):  # Loop over observation_horizon
                    frame = cam_image[:, t]  # (B, C, H, W)
                    cam_features = nets['policy']['backbones'][cam_id](frame)
                    pool_features = nets['policy']['pools'][cam_id](cam_features)
                    pool_features = torch.flatten(pool_features, start_dim=1)
                    out_features = nets['policy']['linears'][cam_id](pool_features)  # (B, feature_dim)
                    temporal_features.append(out_features)
                
                # Stack temporal features: (B, T, feature_dim)
                temporal_features = torch.stack(temporal_features, dim=1)
                all_features.append(temporal_features)
            #################################################################################

            # # 拼接观测条件 / Concatenate observation conditions
            # obs_cond = torch.cat(all_features + [qpos], dim=1)

            ############################ ADDED BELOW ############################
            # qpos shape: (B, observation_horizon, 14)
            # all_features: list of (B, observation_horizon, feature_dim)
            # Need to flatten temporal dimension
            qpos_flat = qpos.reshape(B, -1)  # (B, observation_horizon * 14)
            features_flat = [f.reshape(B, -1) for f in all_features]  # Flatten each
            obs_cond = torch.cat(features_flat + [qpos_flat], dim=1)
            # Result: (B, obs_dim * observation_horizon)
            ######################################################################

            # 初始化动作为高斯噪声 / Initialize actions as Gaussian noise
            noisy_action = torch.randn((B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action
            print('naction_dim:', naction.shape)

            # 设置调度器的时间步长 / Set the scheduler timesteps
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            # 逆扩散过程，逐步去除噪声 / Perform reverse diffusion to gradually remove noise
            for k in self.noise_scheduler.timesteps:
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction, timestep=k, global_cond=obs_cond
                )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample

            return naction

    # 序列化模型，便于保存 / Serialize the model for saving
    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    # 反序列化模型，便于加载 / Deserialize the model for loading
    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status

    # 定义 KL 散度函数，用于计算两个分布之间的差异
    # Define the KL divergence function to compute the difference between two distributions
    def kl_divergence(mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        # 计算 KL 散度 / Calculate the KL divergence
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)  # 总的 KL 散度 / Total KL divergence
        dimension_wise_kld = klds.mean(0)  # 每个维度的平均 KL 散度 / Dimension-wise average KL divergence
        mean_kld = klds.mean(1).mean(0, True)  # 平均 KL 散度 / Mean KL divergence

        return total_kld, dimension_wise_kld, mean_kld
"""
Offline buffer class using dppo's stitched sequence approach.

Converts d3il trajectory datasets into a flat buffer format suitable for offline RL.
"""

import numpy as np
import torch
import logging
from torch.utils.data import Dataset

from d3il.environments.dataset.aligning_dataset import Aligning_Dataset
from d3il.environments.dataset.pushing_dataset import Pushing_Dataset
from d3il.environments.dataset.sorting_dataset import Sorting_Dataset
from d3il.environments.dataset.stacking_dataset import Stacking_Dataset

log = logging.getLogger(__name__)


class OfflineBuffer(Dataset):
    """
    Offline RL buffer using dppo's stitched sequence approach.
    
    Loads d3il datasets and stitches all trajectories together:
        states:  [----traj 1----][----traj 2----] ... [----traj N----]
        actions: [----traj 1----][----traj 2----] ... [----traj N----]
    
    Trajectory boundaries are tracked via traj_lengths array.
    """
    
    DATASET_MAP = {
        "aligning": (Aligning_Dataset, {"obs_dim": 20, "action_dim": 3, "max_len_data": 512}),
        "pushing": (Pushing_Dataset, {"obs_dim": 10, "action_dim": 2, "max_len_data": 512}),
        "sorting": (Sorting_Dataset, {"obs_dim": 10, "action_dim": 2, "max_len_data": 600}),
        "stacking": (Stacking_Dataset, {"obs_dim": 20, "action_dim": 8, "max_len_data": 1000}),
    }
    
    def __init__(
        self,
        dataset_name: str = "aligning",
        data_path: str = None,
        device: str = "cuda:0",
        normalize: bool = True,
    ):
        if dataset_name not in self.DATASET_MAP:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(self.DATASET_MAP.keys())}")
        
        self.device = device
        self.dataset_name = dataset_name
        self.normalize = normalize
        
        # Load d3il dataset with default configs
        dataset_cls, default_cfg = self.DATASET_MAP[dataset_name]
        d3il_dataset = dataset_cls(
            data_directory=data_path,
            device="cpu",  # load to CPU first, then move
            **default_cfg,
        )
        
        # Convert to stitched sequence format (dppo style)
        self._build_stitched_buffer(d3il_dataset)
        
        log.info(f"Loaded {dataset_name} dataset: {self.num_episodes} episodes, {len(self)} transitions")
        log.info(f"States shape: {self.states.shape}, Actions shape: {self.actions.shape}")
    
    def _build_stitched_buffer(self, d3il_dataset):
        """Convert d3il trajectory format to dppo's stitched sequence format."""
        observations = d3il_dataset.observations  # (B, T, obs_dim)
        actions = d3il_dataset.actions            # (B, T, action_dim)
        masks = d3il_dataset.masks                # (B, T)
        
        self.obs_dim = observations.shape[2]
        self.action_dim = actions.shape[2]
        
        # Extract trajectory lengths from masks
        traj_lengths = []
        states_list = []
        actions_list = []
        
        for ep in range(masks.shape[0]):
            T = int(masks[ep].sum().item())
            traj_lengths.append(T)
            states_list.append(observations[ep, :T].numpy())
            actions_list.append(actions[ep, :T].numpy())
        
        self.traj_lengths = np.array(traj_lengths)
        self.num_episodes = len(traj_lengths)
        
        # Stitch all trajectories together
        states_all = np.concatenate(states_list, axis=0)
        actions_all = np.concatenate(actions_list, axis=0)
        
        # Compute normalization stats (dppo style: normalize to [-1, 1])
        if self.normalize:
            self.obs_min = states_all.min(axis=0)
            self.obs_max = states_all.max(axis=0)
            self.action_min = actions_all.min(axis=0)
            self.action_max = actions_all.max(axis=0)
            
            states_all = 2 * (states_all - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 1
            actions_all = 2 * (actions_all - self.action_min) / (self.action_max - self.action_min + 1e-6) - 1
        
        # Move to device
        self.states = torch.from_numpy(states_all).float().to(self.device)
        self.actions = torch.from_numpy(actions_all).float().to(self.device)
        
        # Build rewards and terminals (d3il doesn't have these, so zeros)
        self.rewards = torch.zeros(len(self.states), device=self.device)
        self.terminals = torch.zeros(len(self.states), device=self.device)
        
        # Mark episode boundaries
        cumsum = np.cumsum(self.traj_lengths)
        self.terminals[cumsum - 1] = 1.0  # last step of each episode
        
        # Build valid transition indices (exclude last step of each episode)
        self._build_indices()
    
    def _build_indices(self):
        """Build indices for sampling transitions, respecting episode boundaries."""
        indices = []
        cur_idx = 0
        for traj_len in self.traj_lengths:
            # Include all steps except the last one (no valid next_state)
            indices.extend(range(cur_idx, cur_idx + traj_len - 1))
            cur_idx += traj_len
        self.valid_indices = np.array(indices)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Return a single transition (s, a, r, s', done)."""
        i = self.valid_indices[idx]
        return {
            "state": self.states[i],
            "action": self.actions[i],
            "reward": self.rewards[i],
            "next_state": self.states[i + 1],
            "done": self.terminals[i + 1],
        }
    
    def sample(self, batch_size: int):
        """Sample a random batch of transitions."""
        indices = np.random.choice(len(self.valid_indices), size=batch_size, replace=False)
        batch_indices = self.valid_indices[indices]
        
        return {
            "states": self.states[batch_indices],
            "actions": self.actions[batch_indices],
            "rewards": self.rewards[batch_indices],
            "next_states": self.states[batch_indices + 1],
            "dones": self.terminals[batch_indices + 1],
        }
    
    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Convert normalized action back to original scale."""
        if not self.normalize:
            return action
        action_min = torch.tensor(self.action_min, device=action.device)
        action_max = torch.tensor(self.action_max, device=action.device)
        return (action + 1) / 2 * (action_max - action_min) + action_min
    
    def unnormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Convert normalized state back to original scale."""
        if not self.normalize:
            return state
        obs_min = torch.tensor(self.obs_min, device=state.device)
        obs_max = torch.tensor(self.obs_max, device=state.device)
        return (state + 1) / 2 * (obs_max - obs_min) + obs_min
    
    def sample_initial_states(self, num_states: int = None) -> torch.Tensor:
        """
        Sample initial states from episode starts.
        
        Useful for OPE methods like AM-Q that need to start virtual rollouts
        from real initial state distributions.
        """
        n = num_states if num_states else self.num_episodes
        
        # Compute starting index of each episode
        episode_starts = np.concatenate([[0], np.cumsum(self.traj_lengths)[:-1]])
        
        # Sample random episodes
        selected_eps = np.random.choice(len(episode_starts), size=n, replace=True)
        init_indices = episode_starts[selected_eps]
        
        return self.states[init_indices]

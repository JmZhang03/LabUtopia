import os
import numpy as np
import h5py
import torch
import copy
from typing import Dict
import glob
from policy.dataset.base_dataset import BaseImageDataset
from policy.model.common.normalizer import LinearNormalizer
from policy.model.common.normalizer import SingleFieldLinearNormalizer
from policy.common.normalize_util import get_image_range_normalizer

class DPImageDataset(BaseImageDataset):
    def __init__(self, 
                 shape_meta,
                 dataset_path: str,
                 camera_names: list,
                 horizon: int = None,
                 pad_before: int = None,
                 pad_after: int = None,
                 n_obs_steps: int = None,
                 n_latency_steps: int = None,
                 use_cache: bool = True,
                 seed: int = 42,
                 val_ratio: float = 0.00,
                 delta_action: bool = False,
                 in_memory: bool = True):
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.shape_meta = shape_meta
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.use_cache = use_cache
        self.seed = seed
        self.val_ratio = val_ratio
        self.delta_action = delta_action
        self.episode_map = []
        self.in_memory = in_memory
        self.camera_names = camera_names
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        
        # only support multi-h5 episodes
        h5_files = sorted(glob.glob(os.path.join(dataset_path, "*.h5")))
        assert len(h5_files) > 0, f"No h5 files found in {dataset_path}"

        self.h5_file = {}
        self.episode_map = []
        self.memory_data = {} if self.in_memory else None

        for h5_path in h5_files:
            episode_name = os.path.splitext(os.path.basename(h5_path))[0]
            h5_file = h5py.File(h5_path, 'r')

            self.h5_file[episode_name] = h5_file

            n_frames = h5_file['actions'].shape[0]
            self.episode_map.append((episode_name, n_frames))

            if self.in_memory:
                self.memory_data[episode_name] = {'actions': h5_file['actions'][:], 
                                                  'agent_pose': h5_file['agent_pose'][:]}
                for cam_name in self.camera_names:
                    if cam_name in h5_file:
                        self.memory_data[episode_name][cam_name] = h5_file[cam_name][:]
                    else:
                        print(f"Warning: Camera {cam_name} not found in {episode_name}")

        self.episode_ids = [x[0] for x in self.episode_map]
        
        self.sequences = []
        for episode_name, n_frames in self.episode_map:
            total_steps = n_frames

            # for start_idx in range(total_steps):
            for start_idx in range(max(0, total_steps - self.horizon + 1)):
                self.sequences.append((episode_name, start_idx))

    def __len__(self) -> int:
        return len(self.sequences)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train = False
        if self.in_memory:
            val_set.memory_data = self.memory_data
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.get_all_actions().numpy())
        all_poses = []
        for episode_name, _ in self.episode_map:
            if self.in_memory:
                poses = self.memory_data[episode_name]['agent_pose'].astype(np.float32)
            else:
                episode = self.h5_file[episode_name]
                poses = episode['agent_pose'][:].astype(np.float32)
            all_poses.append(poses)
        all_poses = np.concatenate(all_poses, axis=0)
        normalizer['agent_pose'] = SingleFieldLinearNormalizer.create_fit(all_poses)
        for cam_name in self.camera_names:
            normalizer[cam_name] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        all_actions = []
        for episode_name, _ in self.episode_map:
            if self.in_memory:
                actions = torch.from_numpy(self.memory_data[episode_name]['actions'].astype(np.float32))
            else:
                episode = self.h5_file[episode_name]
                actions = torch.from_numpy(episode['actions'][:].astype(np.float32))
            all_actions.append(actions)
        return torch.cat(all_actions, dim=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_name, start_idx = self.sequences[idx]
        obs_start_idx = start_idx
        obs_end_idx = start_idx + self.n_obs_steps
        action_start_idx = obs_start_idx
        action_end_idx = action_start_idx + self.horizon

        episode = self.memory_data[episode_name] if self.in_memory else self.h5_file[episode_name]
        
        cam_obs_dict = {}
        for cam_name in self.camera_names:
            cam_obs = episode[cam_name][obs_start_idx:obs_end_idx]
            if cam_obs.shape[1] == 1:
                cam_obs = np.repeat(cam_obs, 3, axis=1)
            cam_obs = torch.from_numpy(cam_obs).float() / 255.0
            cam_obs_dict[cam_name] = cam_obs

        robot_eef_obs = episode['agent_pose'][obs_start_idx:obs_end_idx]
        actions = episode['agent_pose'][action_start_idx:action_end_idx]
        robot_eef_obs = torch.from_numpy(robot_eef_obs).float()
        actions = torch.from_numpy(actions).float()

        obs_data = {'agent_pose': robot_eef_obs}
        obs_data.update(cam_obs_dict)

        return {
            'obs': obs_data,
            'action': actions,
        }

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None and isinstance(self.h5_file, h5py.File):
            self.h5_file.close()
            
    @staticmethod
    def collate_fn(batch):

        cam_keys = [k for k in batch[0]['obs'].keys() if 'rgb' in k]
        obs_dict = {'agent_pose': torch.stack([item['obs']['agent_pose'] for item in batch])}
        for cam_key in cam_keys:
            obs_dict[cam_key] = torch.stack([item['obs'][cam_key] for item in batch])
        actions = torch.stack([item['action'] for item in batch])
        
        return {
            'obs': obs_dict,
            'action': actions,
        }

import torch
from torch.utils.data import DataLoader
def main():
    
    dataset_path = ''  
    shape_meta = {'camera_1': (3, 480, 480), 'camera_2': (3, 480, 480), 'agent_pose': (8,)}  
    horizon = 8
    n_obs_steps = 3
    n_latency_steps = 0
    batch_size = 1

    dataset = DPImageDataset(
        shape_meta=shape_meta,
        dataset_path=dataset_path,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_latency_steps=n_latency_steps,
        use_cache=True,
        seed=42,
        val_ratio=0.1,
        in_memory=True
    )

    val_dataset = dataset.get_validation_dataset()

    print(f": {len(dataset)}")
    print(f": {len(val_dataset)}")
    
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=DPImageDataset.collate_fn, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=DPImageDataset.collate_fn, shuffle=False)

    for batch in train_loader:
        print(":")
        print("Camera 1:", batch['obs']['camera_1_rgb'].shape)  # [B, T, 3, H, W]
        print("Camera 2:", batch['obs']['camera_2_rgb'].shape)  # [B, T, 3, H, W]
        print("Agent Pose:", batch['obs']['agent_pose'].shape)  # [B, T, 2]
        print("Actions:", batch['action'].shape)  # [B, T, 2]
        print("Agent Pose:", batch['obs']['agent_pose'])
        print(batch['action'])
        break

if __name__ == "__main__":
    main()
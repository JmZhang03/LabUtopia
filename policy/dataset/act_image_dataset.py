import copy
import glob
import os
import numpy as np
import h5py
import torch
from typing import Dict
from policy.dataset.base_dataset import BaseImageDataset
from policy.model.common.normalizer import LinearNormalizer
from policy.model.common.normalizer import SingleFieldLinearNormalizer
from policy.common.normalize_util import get_image_range_normalizer
from torch.nn.utils.rnn import pad_sequence

class ACTImageDataset(BaseImageDataset):
    def __init__(self, 
                 shape_meta,
                 dataset_path: str,
                 camera_names: list,
                 seed: int = 42,
                 horizon: int = None,
                 n_obs_steps: int = None,
                 val_ratio: float = 0.00,
                 in_memory: bool = True):
        self.dataset_path = dataset_path
        self.shape_meta = shape_meta
        self.seed = seed
        self.val_ratio = val_ratio
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

            for start_idx in range(total_steps):
                self.sequences.append((episode_name, start_idx))

    def __len__(self) -> int:
        return len(self.sequences)

    def get_all_actions(self) -> torch.Tensor:
        all_actions = []
        for episode_name in self.episode_ids:
            if self.in_memory:
                actions = torch.from_numpy(self.memory_data[episode_name]['actions'].astype(np.float32))
            else:
                episode = self.h5_file[episode_name]
                actions = torch.from_numpy(episode['actions'][:].astype(np.float32))
            all_actions.append(actions)
        return torch.cat(all_actions, dim=0)
    
    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.get_all_actions().numpy())
        all_poses = []
        for episode_name in self.episode_ids:
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
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train = False
        if self.in_memory:
            val_set.memory_data = self.memory_data
        return val_set
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode_name, start_idx = self.sequences[index]
        episode = self.memory_data[episode_name] if self.in_memory else self.h5_file[episode_name]
        
        original_action_shape = episode['actions'].shape
        obs_start_idx = start_idx
        obs_end_idx = start_idx + self.n_obs_steps
        action_start_idx = obs_end_idx
        episode_len = original_action_shape[0]
        qpos = episode['agent_pose'][obs_start_idx]
        
        image_dict = dict()
        for cam_name in self.camera_names:
            img_data = episode[cam_name][obs_start_idx]
            image_dict[cam_name] = torch.from_numpy(img_data).float() / 255.0
        
        action = episode['agent_pose'][action_start_idx:]
        action_len = episode_len - action_start_idx
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1
        
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        obs_dict = {'agent_pose': qpos_data}
        obs_dict.update(image_dict)

        return {
            'obs': obs_dict,
            'action': action_data,
            'is_pad': is_pad
        }
    
    @staticmethod
    def collate_fn(batch):
        obs_keys = batch[0]['obs'].keys()
        stacked_obs = {}
        for key in obs_keys:
            items = [item['obs'][key] for item in batch]
            # qpos: list of (D,) -> stack -> (B, D)
            # image: list of (H, W, C) -> stack -> (B, H, W, C)
            stacked_obs[key] = torch.stack(items, dim=0)
            
        actions = [item['action'] for item in batch]   # list of (L, D)
        is_pad = [item['is_pad'] for item in batch]    # list of (L,)

        padded_actions = pad_sequence(actions, batch_first=True)  # -> (B, L_max, D)
        padded_is_pad = pad_sequence(is_pad, batch_first=True)    # -> (B, L_max)

        return {
            'obs': stacked_obs,
            'action': padded_actions,
            'is_pad': padded_is_pad
        }


from torch.utils.data import DataLoader
def main():
    
    dataset_path = ''  
    shape_meta = {'camera_1': (3, 256, 256), 'camera_2': (3, 256, 256), 'agent_pose': (8,)}  

    
    dataset = ACTImageDataset(
        shape_meta=shape_meta,
        dataset_path=dataset_path,
        seed=42,
        val_ratio=0.1,
        n_obs_steps=1,
        horizon=60,
        in_memory=True
    )

    val_dataset = dataset.get_validation_dataset()
    
    print(f": {len(dataset)}")
    print(f": {len(val_dataset)}")
    batch_size = 1  # Assuming a batch size, adjust as needed
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=ACTImageDataset.collate_fn, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=ACTImageDataset.collate_fn, shuffle=False)

    
    for batch in train_loader:
        print(":")
        print("Camera 1:", batch['obs']['camera_1_rgb'].shape)  # [B, T, 3, H, W]
        print("Camera 2:", batch['obs']['camera_2_rgb'].shape)  # [B, T, 3, H, W]
        print("Camera 3:", batch['obs']['camera_3_rgb'].shape)  # [B, T, 3, H, W]
        print("Agent Pose:", batch['obs']['agent_pose'].shape)  # [B, T, 2]
        print("Actions:", batch['action'].shape)  # [B, T, 2]
        camera_1_tensor = batch['obs']['camera_1_rgb']

        max_value = torch.max(camera_1_tensor)
        min_value = torch.min(camera_1_tensor)
        mean_value = torch.mean(camera_1_tensor)
        
        print(f": {max_value.item()}")
        print(f": {min_value.item()}")
        print(f": {mean_value.item()}")
        print("Agent Pose:", batch['obs']['agent_pose'])
        print(batch['action'])
        break
    

if __name__ == "__main__":
    main()

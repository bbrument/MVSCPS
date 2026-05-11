import logging
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

logger = logging.getLogger(__name__)


class LightingOutput(NamedTuple):
    intensity: torch.Tensor                    # (N_rays, 3)
    light_type: str                            # 'directional' or 'point'
    direction: Optional[torch.Tensor] = None   # (N_rays, 3) unit vector, only for directional
    position: Optional[torch.Tensor] = None    # (N_rays, 3) camera space, only for point


class LightingParameters(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.light_type = config.get('light_type', 'directional')
        self.per_image_light = config.get('per_image_light', False)

        # Read light indices used during training
        with open(config.view_light_index_fname_train, 'r') as f:
            view_light_indices = f.read().splitlines()

        if self.per_image_light:
            self.train_num_lights = len(view_light_indices)
            self.train_light_indices = np.arange(self.train_num_lights)
            logger.info(f"per_image_light=True: {self.train_num_lights} independent light parameters")
        else:
            self.train_light_indices = np.unique([int(idx.split("L")[-1]) for idx in view_light_indices])
            self.train_num_lights = len(self.train_light_indices)

        # Initialize light directions or positions depending on light_type
        if self.light_type == 'directional':
            if not self.per_image_light and config.light_dir_file != "None":
                light_dirs = np.loadtxt(config.light_dir_file)  # (N_L, 3)
                light_dirs[..., [1, 2]] = -light_dirs[..., [1, 2]]
                if not config.use_gt_light:
                    light_dirs[self.train_light_indices] = np.array(config.init_light_dir)
            else:
                light_dirs = np.tile(np.array(config.init_light_dir), (self.train_num_lights, 1))

            self.light_dir = nn.Parameter(torch.tensor(light_dirs, dtype=torch.float32), requires_grad=True)
        else:
            light_pos_file = config.get('light_pos_file', 'None')
            if not self.per_image_light and light_pos_file != "None":
                data = np.load(light_pos_file)
                if 'light_positions' in data:
                    light_pos = data['light_positions']
                else:
                    keys = sorted([k for k in data.keys() if k.startswith('view_')])
                    light_pos = np.concatenate([data[k] for k in keys], axis=0)
            else:
                init_pos = np.array(config.get('init_light_pos', [0, 0, -1.0]))
                light_pos = np.tile(init_pos, (self.train_num_lights, 1))

            self.light_pos = nn.Parameter(torch.tensor(light_pos, dtype=torch.float32), requires_grad=True)

        # Initialize light intensities
        if not self.per_image_light and config.light_int_file != "None":
            light_intensity = np.loadtxt(config.light_int_file)  # (N_L, 3)
            if not config.use_gt_light:
                light_intensity[self.train_light_indices] = np.array(config.init_intensity)
        else:
            n_lights = self.train_num_lights
            if not self.per_image_light and self.light_type == 'directional' and config.light_dir_file != "None":
                n_lights = len(np.loadtxt(config.light_dir_file))
            light_intensity = np.ones((n_lights, 3)) * np.array(config.init_intensity)

        self.intensity = nn.Parameter(torch.tensor(light_intensity, dtype=torch.float32), requires_grad=True)  # (N_L, 3)

        # Freeze lighting parameters if requested
        if config.get('freeze_light', False):
            if self.light_type == 'directional':
                self.light_dir.requires_grad_(False)
            else:
                self.light_pos.requires_grad_(False)
            self.intensity.requires_grad_(False)
            logger.info("Lighting parameters frozen (freeze_light=True)")

    def forward(self, rays_light_indices):
        """
        Args:
            rays_light_indices: (N_rays,) LongTensor
        Returns:
            LightingOutput with intensity and either direction or position
        """
        rays_light_intensity = self.intensity[rays_light_indices]

        if self.light_type == 'directional':
            rays_light_dir = Fn.normalize(self.light_dir, p=2, dim=-1)[rays_light_indices]
            return LightingOutput(
                intensity=rays_light_intensity,
                light_type='directional',
                direction=rays_light_dir,
            )
        else:
            rays_light_pos = self.light_pos[rays_light_indices]
            return LightingOutput(
                intensity=rays_light_intensity,
                light_type='point',
                position=rays_light_pos,
            )

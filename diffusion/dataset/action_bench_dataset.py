"""ActionBench dataset for UniSkill training.

Loads demonstration data from the ActionBench ManiSkill format:
    <data_path>/
        0000/
            <timestamp>.h5        # obs (sensor_data/base_camera/rgb), actions
            <timestamp>.json      # env metadata (env_id, env_kwargs, episodes)
            0.mp4                 # rendered video
        0001/
            ...

Each .h5 file contains one trajectory (traj_0) with RGB observations stored at
traj_0/obs/sensor_data/base_camera/rgb as uint8 arrays of shape (T, H, W, 3).

The dataset presents frame pairs (current, future) sampled from demonstrations,
matching the interface expected by UniSkill's IDM training pipeline.
"""

import json
import os

import h5py
import numpy as np
from PIL import Image

from .base_dataset import BaseDataset


class ActionBenchDataset(BaseDataset):
    def __init__(
        self,
        data_path: str = "/workspace/datasets/action_bench",
        camera_key: str = "base_camera",
        **kwargs,
    ):
        kwargs["min_predict_future_horizon"] = 10
        kwargs["max_predict_future_horizon"] = 30
        self.camera_key = camera_key
        super().__init__(data_path, **kwargs)

    def _prepare_data(self, data_path):
        """Discover all demo directories and build the image_pair list.

        Each demo directory contains an .h5 file with trajectory data.
        We read the RGB observation length from the file to populate
        the image_pair list used by BaseDataset.__getitem__.
        """
        demos = []
        for entry in sorted(os.listdir(data_path)):
            demo_dir = os.path.join(data_path, entry)
            if not os.path.isdir(demo_dir):
                continue

            # Find the main .h5 file (not the .state. variant)
            h5_files = [
                f for f in os.listdir(demo_dir)
                if f.endswith(".h5") and ".state." not in f
            ]
            if not h5_files:
                continue

            h5_path = os.path.join(demo_dir, h5_files[0])

            # Read trajectory length from the H5 file
            try:
                with h5py.File(h5_path, "r") as f:
                    rgb_key = f"traj_0/obs/sensor_data/{self.camera_key}/rgb"
                    if rgb_key not in f:
                        continue
                    vid_len = f[rgb_key].shape[0]
            except Exception:
                continue

            demos.append({"path": h5_path, "length": vid_len})

        # Train/val split: 90/10 by sorted order
        total = len(demos)
        if self.train:
            demos = demos[: int(total * 0.9)]
        else:
            demos = demos[int(total * 0.9) :]

        # Filter out demos shorter than min horizon
        self.image_pair = [
            d for d in demos if d["length"] >= self.min_predict_future_horizon
        ]

    def read_images(self, video_path, prev_idx, next_idx):
        """Load two RGB frames from the HDF5 trajectory.

        Args:
            video_path: Path to the .h5 file.
            prev_idx: Index of the current frame.
            next_idx: Index of the future frame.

        Returns:
            Tuple of (curr_image, next_image) as PIL Images.
        """
        with h5py.File(video_path, "r") as f:
            rgb = f[f"traj_0/obs/sensor_data/{self.camera_key}/rgb"]
            curr_frame = rgb[prev_idx]
            next_frame = rgb[next_idx]

        curr_image = Image.fromarray(curr_frame)
        next_image = Image.fromarray(next_frame)

        return curr_image, next_image

"""ActionBench dataset for UniSkill training (LfO layout).

Loads frame pairs for IDM/FSD training from the LfO sample-data layout:

    <data_path>/
        video_demonstrations/<task>/<demo_id>/
            <timestamp>.h5        # obs (traj_0/obs/sensor_data/<camera>/rgb)
            <timestamp>.json      # env metadata
            0.mp4                 # rendered video
        training_trajectories/    # robot rollouts; NOT used here.

Each ``.h5`` file contains one trajectory (``traj_0``) with RGB observations at
``traj_0/obs/sensor_data/<camera>/rgb`` as uint8 arrays of shape (T, H, W, 3).
The dataset presents pairs of frames (current, future) sampled from these
videos, matching the interface expected by UniSkill's IDM training pipeline.

Phase 1 (FSD/ISD) trains on the **video** side of the LfO split; only
``video_demonstrations/`` is read here. ``training_trajectories/`` is reserved
for skill extraction + policy training in phases 2/3 of the pipeline.
"""

import os

import h5py
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
        """Walk ``<data_path>/video_demonstrations/<task>/<demo_id>/`` and
        build the image_pair list.

        Demos are flattened across tasks (sorted by ``(task, demo_id)``) so the
        90/10 train/val split is deterministic. For each demo we read the RGB
        trajectory length up front; ``BaseDataset.__getitem__`` then samples
        random ``(prev_idx, next_idx)`` pairs and pulls frames via
        :meth:`read_images`.
        """
        # Be tolerant of an old-style flat layout to keep older configs alive.
        video_root = os.path.join(data_path, "video_demonstrations")
        if not os.path.isdir(video_root):
            video_root = data_path

        demos = []
        for task_entry in sorted(os.listdir(video_root)):
            task_dir = os.path.join(video_root, task_entry)
            if not os.path.isdir(task_dir):
                continue
            for demo_entry in sorted(os.listdir(task_dir)):
                demo_dir = os.path.join(task_dir, demo_entry)
                if not os.path.isdir(demo_dir):
                    continue

                h5_files = [
                    f for f in sorted(os.listdir(demo_dir))
                    if f.endswith(".h5") and ".state." not in f
                ]
                if not h5_files:
                    continue
                h5_path = os.path.join(demo_dir, h5_files[0])

                try:
                    with h5py.File(h5_path, "r") as f:
                        rgb_key = f"traj_0/obs/sensor_data/{self.camera_key}/rgb"
                        if rgb_key not in f:
                            continue
                        vid_len = f[rgb_key].shape[0]
                except Exception:
                    continue

                demos.append({"path": h5_path, "length": vid_len})

        total = len(demos)
        if self.train:
            demos = demos[: int(total * 0.9)]
        else:
            demos = demos[int(total * 0.9) :]

        # Filter out demos shorter than min horizon (we'd never sample a valid
        # pair from them otherwise).
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

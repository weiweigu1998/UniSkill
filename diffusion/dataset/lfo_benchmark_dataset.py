"""LfO Benchmark dataset for UniSkill — mixes robot trajectories and human videos.

Reads two heterogeneous sources from the action_understanding_bench LfO layout:

    Robot:  <data_path>/h5_training_trajectories/<task>/<demo>/<ts>.h5
            key = traj_0/obs/sensor_data/<camera>/rgb,  shape (T, 512, 512, 3) uint8
            cameras: {base, hand, left, overhead, right}_camera

    Human:  <data_path>/raw_video_demonstrations/<distract>/<task>/<demo>/<camera>.mp4
            cameras: {egocentric, front, left, right} at 720x1280 uint8

``BaseDataset.image_transforms`` resizes everything to ``resolution`` so the
two source resolutions are fine to mix. Selection is config-driven so callers
can pick a task list, a per-source camera list, and a per-source demo cap.
The 90/10 train/val split is applied **independently per source** so the
validation set also has both robot and human frames.

The IDM/FSD training loop expects ``self.image_pair`` to be a list of dicts
with ``path`` and ``length``. Here each entry additionally carries ``source``
(``"robot"`` or ``"human"``) and ``camera``; the overridden ``__getitem__``
threads the full entry into ``read_images`` so it can dispatch on source.
"""

import os
import random

import h5py
from PIL import Image
from decord import VideoReader, cpu

from .base_dataset import BaseDataset

#: Subdirectory holding processed sensor_data-mode robot trajectories.
ROBOT_SUBDIR = "h5_training_trajectories"
#: Subdirectory holding raw human demonstration videos.
HUMAN_SUBDIR = "raw_video_demonstrations"
#: Human-demo distraction sub-bucket: True picks ``with_distraction/``.
HUMAN_DISTRACT_DIRS = {True: "with_distraction", False: "without_distraction"}

#: Cameras available for each source — used only for friendlier error messages.
ROBOT_CAMERAS = ("base_camera", "hand_camera", "left_camera", "overhead_camera", "right_camera")
HUMAN_CAMERAS = ("egocentric", "front", "left", "right")


class LfOBenchmarkDataset(BaseDataset):
    """Frame-pair dataset over (robot trajectories ∪ human demonstration videos).

    Args:
        data_path: Root holding ``h5_training_trajectories/`` and ``raw_video_demonstrations/``.
        tasks: Tasks to include. ``None`` = intersection of tasks present in both sources.
        robot_cameras: Cameras to load from robot trajectories.
        human_cameras: Cameras to load from human videos.
        num_robot_demos_per_task: Cap on robot demos per task (after sort). ``None`` = all.
        num_human_demos_per_task: Cap on human demos per task (after sort). ``None`` = all.
        human_distract: ``True`` for ``with_distraction/`` videos, ``False`` for ``without_distraction/``.
        **kwargs: Forwarded to :class:`BaseDataset` (``train``, ``resolution``,
            ``idm_resolution``, ``depth_processor``, horizon args, ...).
    """

    def __init__(
        self,
        data_path: str,
        tasks=None,
        robot_cameras=("base_camera",),
        human_cameras=("front",),
        num_robot_demos_per_task=None,
        num_human_demos_per_task=None,
        human_distract: bool = False,
        **kwargs,
    ):
        self.tasks = list(tasks) if tasks is not None else None
        self.robot_cameras = list(robot_cameras)
        self.human_cameras = list(human_cameras)
        self.num_robot_demos_per_task = num_robot_demos_per_task
        self.num_human_demos_per_task = num_human_demos_per_task
        self.human_distract = bool(human_distract)
        # Default horizon range — robot trajectories are 139-459 frames, human videos
        # are ~30-100, so 10-30 fits both. Callers can override via kwargs.
        kwargs.setdefault("min_predict_future_horizon", 10)
        kwargs.setdefault("max_predict_future_horizon", 30)
        super().__init__(data_path, **kwargs)

    def _prepare_data(self, data_path):
        robot_root = os.path.join(data_path, ROBOT_SUBDIR)
        human_root = os.path.join(data_path, HUMAN_SUBDIR, HUMAN_DISTRACT_DIRS[self.human_distract])

        # Resolve task set: intersection of tasks present in both sources by default.
        robot_tasks = _list_subdirs(robot_root)
        human_tasks = _list_subdirs(human_root)
        if self.tasks is None:
            tasks = sorted(set(robot_tasks) & set(human_tasks))
        else:
            tasks = list(self.tasks)
        if not tasks:
            raise ValueError(
                f"No tasks resolved. robot_root={robot_root!r} has {sorted(robot_tasks)}, "
                f"human_root={human_root!r} has {sorted(human_tasks)}."
            )

        robot_entries = self._scan_robot(robot_root, tasks)
        human_entries = self._scan_human(human_root, tasks)

        # Per-source 90/10 train/val split so val has both sources.
        self.image_pair = _split(robot_entries, self.train) + _split(human_entries, self.train)

    def _scan_robot(self, robot_root, tasks):
        out = []
        for task in tasks:
            task_dir = os.path.join(robot_root, task)
            if not os.path.isdir(task_dir):
                continue
            demo_names = sorted(d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d)))
            if self.num_robot_demos_per_task is not None:
                demo_names = demo_names[: self.num_robot_demos_per_task]
            for demo in demo_names:
                demo_dir = os.path.join(task_dir, demo)
                h5_files = [f for f in sorted(os.listdir(demo_dir)) if f.endswith(".h5") and ".state." not in f]
                if not h5_files:
                    continue
                h5_path = os.path.join(demo_dir, h5_files[0])
                try:
                    with h5py.File(h5_path, "r") as f:
                        per_cam_len = {}
                        for cam in self.robot_cameras:
                            key = f"traj_0/obs/sensor_data/{cam}/rgb"
                            if key in f:
                                per_cam_len[cam] = int(f[key].shape[0])
                except (OSError, KeyError):
                    continue
                for cam, T in per_cam_len.items():
                    if T < self.min_predict_future_horizon:
                        continue
                    out.append({"path": h5_path, "length": T, "source": "robot", "camera": cam})
        return out

    def _scan_human(self, human_root, tasks):
        out = []
        for task in tasks:
            task_dir = os.path.join(human_root, task)
            if not os.path.isdir(task_dir):
                continue
            demo_names = sorted(d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d)))
            if self.num_human_demos_per_task is not None:
                demo_names = demo_names[: self.num_human_demos_per_task]
            for demo in demo_names:
                demo_dir = os.path.join(task_dir, demo)
                for cam in self.human_cameras:
                    mp4 = os.path.join(demo_dir, f"{cam}.mp4")
                    if not os.path.isfile(mp4):
                        continue
                    try:
                        T = len(VideoReader(mp4, ctx=cpu(0)))
                    except Exception:
                        continue
                    if T < self.min_predict_future_horizon:
                        continue
                    out.append({"path": mp4, "length": int(T), "source": "human", "camera": cam})
        return out

    def __getitem__(self, idx):
        entry = self.image_pair[idx]
        video_len = entry["length"]

        # Same horizon sampling as ``BaseDataset.__getitem__``.
        while True:
            predict_future_horizon = random.randint(
                self.min_predict_future_horizon, self.max_predict_future_horizon
            )
            predict_future_horizon = min(predict_future_horizon, video_len - 1)
            prev_idx = random.randint(0, video_len - predict_future_horizon - 1)
            next_idx = prev_idx + predict_future_horizon
            if next_idx < video_len:
                break

        curr_image, next_image = self.read_images_entry(entry, prev_idx, next_idx)

        idm_curr_image = self.idm_image_transforms(curr_image)
        idm_next_image = self.idm_image_transforms(next_image)
        curr_image = self.image_transforms(curr_image)
        next_image = self.image_transforms(next_image)

        curr_depth_features = self.depth_processor(idm_curr_image, do_rescale=False)["pixel_values"][0]
        next_depth_features = self.depth_processor(idm_next_image, do_rescale=False)["pixel_values"][0]

        if self.train:
            curr_image = self.fdm_normalize(curr_image)
            next_image = self.fdm_normalize(next_image)

        return {
            "curr_images": curr_image,
            "next_images": next_image,
            "idm_curr_images": idm_curr_image,
            "idm_next_images": idm_next_image,
            "curr_depth_features": curr_depth_features,
            "next_depth_features": next_depth_features,
        }

    def read_images_entry(self, entry, prev_idx, next_idx):
        if entry["source"] == "robot":
            with h5py.File(entry["path"], "r") as f:
                rgb = f[f"traj_0/obs/sensor_data/{entry['camera']}/rgb"]
                curr = rgb[prev_idx]
                nxt = rgb[next_idx]
        elif entry["source"] == "human":
            vr = VideoReader(entry["path"], ctx=cpu(0))
            curr = vr[prev_idx].asnumpy()
            nxt = vr[next_idx].asnumpy()
        else:
            raise ValueError(f"Unknown source: {entry['source']!r}")
        return Image.fromarray(curr), Image.fromarray(nxt)


def _list_subdirs(root):
    if not os.path.isdir(root):
        return set()
    return {p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))}


def _split(entries, train):
    """90/10 train/val split. Operates on a flat list sorted by (task, demo, camera)
    via the sort already applied during scan, so the split is deterministic."""
    n = len(entries)
    cut = int(n * 0.9)
    return entries[:cut] if train else entries[cut:]

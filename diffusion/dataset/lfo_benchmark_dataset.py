"""LfO Benchmark dataset for UniSkill — mixes robot trajectories (read from
the new ``postprocessed_robot_trajectories`` per-sample pkl format) with human demonstration
videos (read from the original ``raw_video_demonstrations`` mp4s).

Sources::

    Robot:  <data_path>/postprocessed_robot_trajectories/data/<idx>.pkl  +  meta/{stats.json,
            index.jsonl}                             # one frame per pkl,
                                                      # pre-resized to 224²
            ↑ produced by scripts/process_training_trajectories.py
              with --pi05-samples. Each pkl carries one robot timestep's
              observation/<camera> (HWC uint8) plus state/actions metadata.

    Human:  <data_path>/raw_video_demonstrations/<distract>/<task>/<demo>/<camera>.mp4
            cameras: {egocentric, front, left, right}

Frame pairs (curr, next) are sampled per-demo: for the robot side we group
the sample indices by (task, demo_id) — using ``postprocessed_robot_trajectories/meta/index.jsonl``
— and treat each group as a virtual trajectory of length ``len(group)``.
``read_images_entry`` then loads the curr/next pkls and pulls ``observation/<cam>``
out of each. For the human side we still decode the mp4 with decord exactly
like before, since human demos don't go through postprocessed_robot_trajectories.

This swap drops the dependency on ``h5_training_trajectories/`` entirely: the
UniSkill IDM trainer can run against the same pkl artefacts the Pi0.5 trainer
consumes, and only the timesteps we actually visit per __getitem__ pay disk
I/O (one pickle load per frame, not a whole-trajectory h5 read).

``BaseDataset.image_transforms`` resizes everything to ``resolution`` so the
two source resolutions are fine to mix. Selection is config-driven so callers
can pick a task list, a per-source camera list, and a per-source demo cap.
The 90/10 train/val split is applied **independently per source** so the
validation set also has both robot and human frames.
"""

from __future__ import annotations

import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from decord import VideoReader, cpu

from .base_dataset import BaseDataset

#: Subdirectory holding the new per-sample pkl format.
ROBOT_SAMPLES_SUBDIR = "postprocessed_robot_trajectories"
#: Subdirectory holding raw human demonstration videos.
HUMAN_SUBDIR = "raw_video_demonstrations"
#: Distraction sub-buckets selectable via ``include_video_types``. Filters both
#: the robot index and the human video pool — both sides must come from the
#: same bucket so the IDM doesn't accidentally learn cross-bucket invariants.
VALID_VIDEO_TYPES = ("without_distraction", "with_distraction", "task_sequence")

#: Buckets whose human-demo tree is flat (demos sit directly under the bucket
#: dir, no per-task layer): currently ``task_sequence``.
_FLAT_HUMAN_BUCKETS = frozenset({"task_sequence"})


def _infer_entry_bucket(entry: dict) -> str:
    """Fallback bucket label for legacy index entries lacking a "bucket" field."""
    b = entry.get("bucket")
    if b:
        return b
    if " and " in entry.get("task", ""):
        return "task_sequence"
    return "without_distraction"

#: Cameras available for each source — used only for friendlier error messages.
ROBOT_CAMERAS = ("base_camera", "hand_camera", "left_camera", "overhead_camera", "right_camera")
HUMAN_CAMERAS = ("egocentric", "front", "left", "right")


class LfOBenchmarkDataset(BaseDataset):
    """Frame-pair dataset over (robot trajectories ∪ human demonstration videos).

    Args:
        data_path: Root holding ``postprocessed_robot_trajectories/`` and ``raw_video_demonstrations/``.
        tasks: Tasks to include. ``None`` = intersection of tasks present in both sources.
        robot_cameras: Cameras to read from each robot pkl (each pkl must carry
            ``observation/<cam>`` for every listed camera).
        human_cameras: Cameras to load from human videos.
        num_robot_demos_per_task: Cap on robot demos per task (after sort). ``None`` = all.
        num_human_demos_per_task: Cap on human demos per task **per video type**
            (after sort). ``None`` = all.
        include_video_types: Human-demo distraction buckets to mix in. A list of
            ``{"without_distraction", "with_distraction"}`` — the cap applies per
            bucket, so ``["without_distraction", "with_distraction"]`` with
            ``num_human_demos_per_task=80`` yields 80 demos from each bucket.
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
        include_video_types=("without_distraction",),
        **kwargs,
    ):
        self.tasks = list(tasks) if tasks is not None else None
        self.robot_cameras = list(robot_cameras)
        self.human_cameras = list(human_cameras)
        self.num_robot_demos_per_task = num_robot_demos_per_task
        self.num_human_demos_per_task = num_human_demos_per_task
        self.include_video_types = list(include_video_types)
        bad = [v for v in self.include_video_types if v not in VALID_VIDEO_TYPES]
        if bad or not self.include_video_types:
            raise ValueError(
                f"include_video_types must be a non-empty subset of {VALID_VIDEO_TYPES}; "
                f"got {self.include_video_types!r}."
            )
        # Default horizon range — matches the UniSkill paper's stage-1 recipe
        # (Table 8a: k sampled from [20, 40]). Our shortest human demo is 84
        # frames (without_distraction), well above 40, so this range is safe
        # for every demo we have. Callers can override via kwargs if a smaller
        # range is needed (e.g. for a new bucket with very short videos).
        kwargs.setdefault("min_predict_future_horizon", 20)
        kwargs.setdefault("max_predict_future_horizon", 40)
        super().__init__(data_path, **kwargs)

    # ------------------------------------------------------------------
    # _prepare_data — build self.image_pair across robot + human sources.
    # ------------------------------------------------------------------

    def _prepare_data(self, data_path):
        robot_samples_root = Path(data_path) / ROBOT_SAMPLES_SUBDIR
        human_base = Path(data_path) / HUMAN_SUBDIR

        # robot_index: {bucket: {task: {demo_id: [sample_idx, ...]}}}
        robot_index = self._load_robot_sample_index(robot_samples_root)

        # Per-bucket task selection. For per-task buckets we intersect robot
        # and human task names; for flat buckets (task_sequence) there is no
        # task layer on the human side, so we keep all robot tasks in that
        # bucket and emit human demos directly from the bucket dir.
        per_bucket_tasks: dict[str, list[str]] = {}
        for bucket in self.include_video_types:
            robot_tasks = set(robot_index.get(bucket, {}).keys())
            if bucket in _FLAT_HUMAN_BUCKETS:
                tasks = sorted(robot_tasks) if self.tasks is None else list(self.tasks)
            else:
                human_tasks = _list_subdirs(human_base / bucket)
                if self.tasks is None:
                    tasks = sorted(robot_tasks & human_tasks)
                else:
                    tasks = list(self.tasks)
            per_bucket_tasks[bucket] = tasks

        if not any(per_bucket_tasks.values()):
            raise ValueError(
                f"No tasks resolved for any bucket. robot_samples_root="
                f"{robot_samples_root!r} buckets={list(robot_index.keys())}, "
                f"human_base={human_base!r} video_types={self.include_video_types}."
            )

        robot_entries = self._scan_robot(robot_samples_root, robot_index, per_bucket_tasks)
        human_entries = self._scan_human(human_base, per_bucket_tasks)

        # Per-source 90/10 train/val split so val has both sources.
        self.image_pair = _split(robot_entries, self.train) + _split(human_entries, self.train)

    def _load_robot_sample_index(
        self, robot_samples_root: Path
    ) -> dict[str, dict[str, dict[str, list[int]]]]:
        """Read ``postprocessed_robot_trajectories/meta/index.jsonl`` and group sample idxs by
        ``(bucket, task, demo_id)``. Returns
        ``{bucket: {task: {demo_id: [sorted sample_idx, ...]}}}``.

        Bucket comes from the entry's ``"bucket"`` field (filled in by the
        bucketed preprocessor); legacy entries without it fall back to
        ``_infer_entry_bucket``. Filtering robot demos by bucket is Fix A's
        precondition — without it we'd mix robot frames from a different
        distraction setting into the IDM training set.
        """
        index_path = robot_samples_root / "meta" / "index.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"Expected postprocessed_robot_trajectories meta/index.jsonl at {index_path}. "
                f"Run `scripts/process_training_trajectories.py --pi05-samples` "
                f"first to materialize the per-sample pkl dataset."
            )
        allowed = set(self.include_video_types)
        groups: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        with open(index_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                bucket = _infer_entry_bucket(entry)
                if bucket not in allowed:
                    continue
                groups[bucket][entry["task"]][entry["demo_id"]].append(int(entry["idx"]))
        # Sort the per-demo idx lists so the group represents (t=0, t=1, …).
        for bucket, by_task in groups.items():
            for task, demos in by_task.items():
                for demo_id, idxs in demos.items():
                    demos[demo_id] = sorted(idxs)
        return groups

    def _scan_robot(self, robot_samples_root: Path, index: dict,
                    per_bucket_tasks: dict[str, list[str]]):
        """One ``image_pair`` entry per (bucket, demo, camera). ``path`` is the
        demo's absolute pkl directory, ``length`` is the number of sample pkls
        in it, and ``camera`` selects which ``observation/<cam>`` key to slice
        at __getitem__ time.
        """
        out = []
        data_dir = robot_samples_root / "data"
        for bucket, tasks in per_bucket_tasks.items():
            by_task = index.get(bucket, {})
            for task in tasks:
                demos = by_task.get(task, {})
                demo_ids_sorted = sorted(demos.keys())
                if self.num_robot_demos_per_task is not None:
                    demo_ids_sorted = demo_ids_sorted[: self.num_robot_demos_per_task]
                for demo_id in demo_ids_sorted:
                    sample_idxs = demos[demo_id]
                    T = len(sample_idxs)
                    if T < self.min_predict_future_horizon:
                        continue
                    for cam in self.robot_cameras:
                        out.append({
                            "path": str(data_dir),
                            "length": T,
                            "source": "robot",
                            "camera": cam,
                            "bucket": bucket,
                            "task": task,
                            "demo_id": demo_id,
                            "sample_idxs": sample_idxs,
                        })
        return out

    def _scan_human(self, human_base, per_bucket_tasks: dict[str, list[str]]):
        """One ``image_pair`` entry per (bucket, demo, camera).

        For per-task buckets we walk ``<human_base>/<bucket>/<task>/<demo>/<cam>.mp4``.
        For flat buckets (``task_sequence``) the task layer is absent — demos
        sit directly under the bucket — so we walk ``<human_base>/<bucket>/<demo>/``.
        ``num_human_demos_per_task`` is applied per (bucket, task) for per-task
        buckets, or per bucket for flat buckets.
        """
        out = []
        for video_type, tasks in per_bucket_tasks.items():
            human_root = os.path.join(str(human_base), video_type)
            if video_type in _FLAT_HUMAN_BUCKETS:
                if not os.path.isdir(human_root):
                    continue
                demo_names = sorted(
                    d for d in os.listdir(human_root)
                    if os.path.isdir(os.path.join(human_root, d))
                )
                if self.num_human_demos_per_task is not None:
                    demo_names = demo_names[: self.num_human_demos_per_task]
                for demo in demo_names:
                    demo_dir = os.path.join(human_root, demo)
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
                        out.append({
                            "path": mp4, "length": int(T),
                            "source": "human", "camera": cam,
                            "bucket": video_type,
                        })
                continue
            for task in tasks:
                task_dir = os.path.join(human_root, task)
                if not os.path.isdir(task_dir):
                    continue
                demo_names = sorted(
                    d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))
                )
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
                        out.append({
                            "path": mp4, "length": int(T),
                            "source": "human", "camera": cam,
                            "bucket": video_type,
                        })
        return out

    # ------------------------------------------------------------------
    # __getitem__ / read_images — same shape as BaseDataset's loop but
    # threads the full image_pair entry through ``read_images_entry`` so
    # it can dispatch on ``source``.
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        entry = self.image_pair[idx]
        video_len = entry["length"]

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
            sample_idxs = entry["sample_idxs"]
            data_dir = Path(entry["path"])
            cam = entry["camera"]
            curr_pkl = data_dir / f"{sample_idxs[prev_idx]}.pkl"
            next_pkl = data_dir / f"{sample_idxs[next_idx]}.pkl"
            with open(curr_pkl, "rb") as f:
                curr = pickle.load(f)[f"observation/{cam}"]
            with open(next_pkl, "rb") as f:
                nxt = pickle.load(f)[f"observation/{cam}"]
        elif entry["source"] == "human":
            vr = VideoReader(entry["path"], ctx=cpu(0))
            curr = vr[prev_idx].asnumpy()
            nxt = vr[next_idx].asnumpy()
        else:
            raise ValueError(f"Unknown source: {entry['source']!r}")
        return Image.fromarray(curr), Image.fromarray(nxt)


def _list_subdirs(root):
    root = str(root)
    if not os.path.isdir(root):
        return set()
    return {p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))}


def _split(entries, train):
    """90/10 train/val split. Operates on a flat list already sorted by scan
    order so the split is deterministic across processes."""
    n = len(entries)
    cut = int(n * 0.9)
    return entries[:cut] if train else entries[cut:]

"""Dataset + collate_fn for LoGoPlanner training.

Extends NavDP_Base_Datset to additionally produce the GT signals needed by the
geometry-grounded losses in LoGoPlanner (Peng et al., arxiv 2512.19629).

Per paper §IV.B, supervision comes from:
  - Per-frame camera poses T_c,i (for ExtrinctHead, eq 4)
  - Per-pixel local points P_cam = D(u,v) · K⁻¹ · [u v 1]ᵀ (eq 2/3)
  - Per-pixel world points P_world = T_c,i · P_cam (eq 6)

All geometry GTs are expressed in the chassis frame of the last context step
(paper §IV.C: "world coordinate system is defined with respect to the chassis
frame of the last time step"). This matches what the model predicts.

Additional batch keys beyond NavDP:
  batch_context_rgb       [B, N, Hc, Wc, 3]   N consecutive RGB frames for Pi3
  batch_context_depth     [B, N, Hc, Wc, 1]   matching depth (for depth prior)
  batch_gt_camera_poses   [B, N, 5]           [x, y, z, sinθ, cosθ] relative to current
  batch_gt_local_points   [B, N, Hc, Wc, 3]   unprojected depth in camera frame
  batch_gt_world_points   [B, N, Hc, Wc, 3]   relative to current chassis
  batch_gt_subgoal        [B, 3]              (x, y, θ) goal in current frame

Context image size defaults to (168, 308) — matches the shape used in
NavDP/baselines/logoplanner/policy_network.py __main__ and is divisible by the
ViT patch size 14.

Caveats / design choices flagged for review:
  - Pose encoding is [x, y, z, sinθ, cosθ] (5 dims, matches ExtrinctHead.fc_pose).
    Paper describes 3-DoF (x, y, θ) but the code outputs 5. Change the encoding
    here + the model head together if the paper encoding is preferred.
  - batch_gt_subgoal reuses `point_goal` (final waypoint in current frame).
    If the paper intends a mid-horizon waypoint, change `_compute_subgoal_gt`.
  - Context window is the last N frames ending at `memory_start_choice` with
    stride 1. If a trajectory is too short, earlier slots are zero-padded and
    their GT is masked to zero (so the MSE contribution vanishes).
"""

import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from internnav.dataset.navdp_dataset_lerobot import NavDP_Base_Datset


class LoGoPlanner_Dataset(NavDP_Base_Datset):
    def __init__(
        self,
        root_dirs,
        preload_path=False,
        memory_size=8,
        predict_size=24,
        batch_size=64,
        image_size=224,
        scene_data_scale=1.0,
        trajectory_data_scale=1.0,
        debug=False,
        preload=False,
        random_digit=False,
        prior_sample=False,
        context_size=12,
        context_image_height=168,
        context_image_width=308,
        depth_max=5.0,
        depth_min=0.1,
    ):
        super().__init__(
            root_dirs,
            preload_path=preload_path,
            memory_size=memory_size,
            predict_size=predict_size,
            batch_size=batch_size,
            image_size=image_size,
            scene_data_scale=scene_data_scale,
            trajectory_data_scale=trajectory_data_scale,
            debug=debug,
            preload=preload,
            random_digit=random_digit,
            prior_sample=prior_sample,
        )
        self.context_size = context_size
        self.context_image_height = context_image_height
        self.context_image_width = context_image_width
        self.depth_max = depth_max
        self.depth_min = depth_min

    # ---- Context image/depth loading (aspect-preserving, not squared) ----

    def _resize_preserve(self, arr, h, w, interp=cv2.INTER_LINEAR):
        return cv2.resize(arr, (w, h), interpolation=interp)

    def process_context_image(self, image_path):
        img = self.load_image(image_path)
        img = self._resize_preserve(img, self.context_image_height, self.context_image_width)
        return np.array(img, np.float32) / 255.0

    def process_context_depth(self, depth_path):
        depth = self.load_depth(depth_path) / 10000.0
        depth = self._resize_preserve(depth, self.context_image_height, self.context_image_width, cv2.INTER_NEAREST)
        depth = np.array(depth, np.float32)
        depth[depth > self.depth_max] = 0.0
        depth[depth < self.depth_min] = 0.0
        return depth[:, :, np.newaxis]

    def process_context_window(self, rgb_paths, depth_paths, end_step):
        """Load N consecutive (RGB, depth) frames ending at end_step (inclusive).

        Returns arrays of shape (N, Hc, Wc, 3) and (N, Hc, Wc, 1) plus a mask
        (N,) bool indicating which slots are valid (False → zero-padded).
        """
        N = self.context_size
        Hc, Wc = self.context_image_height, self.context_image_width
        indices = np.arange(end_step - N + 1, end_step + 1)
        pad = (indices < 0).sum()
        indices_valid = indices[pad:]

        rgb = np.zeros((N, Hc, Wc, 3), np.float32)
        depth = np.zeros((N, Hc, Wc, 1), np.float32)
        mask = np.zeros((N,), np.bool_)

        for slot, t in enumerate(indices_valid, start=pad):
            rgb[slot] = self.process_context_image(rgb_paths[t])
            depth[slot] = self.process_context_depth(depth_paths[t])
            mask[slot] = True
        return rgb, depth, mask, indices

    # ---- Memory loader override ----
    # NavDP_RGBD_Backbone hardcodes (memory_size+1)*264 positional embeddings,
    # i.e. a 12x22 patch grid for 168x308 inputs at ViT patch size 14. The base
    # class would produce square image_size x image_size memory frames; override
    # so memory and context share the aspect-preserving 168x308 resize.

    def process_memory(self, rgb_paths, depth_paths, start_step, memory_digit=1):
        memory_index = np.arange(
            start_step - (self.memory_size - 1) * memory_digit, start_step + 1, memory_digit
        )
        outrange_sum = (memory_index < 0).sum()
        memory_index = memory_index[outrange_sum:]
        Hc, Wc = self.context_image_height, self.context_image_width
        context_image = np.zeros((self.memory_size, Hc, Wc, 3), np.float32)
        context_image[outrange_sum:] = np.array(
            [self.process_context_image(rgb_paths[i]) for i in memory_index]
        )
        context_depth = self.process_context_depth(depth_paths[start_step])
        return context_image, context_depth, memory_index

    # ---- Geometry GT computation ----

    def _pose_to_xyzsincos(self, R, T):
        """Encode a ground-plane pose as [x, y, z, sinθ, cosθ].

        θ is the yaw angle between the pose's forward direction and the current
        frame's x-axis. Using a 2D unit vector for yaw (sin, cos) avoids the
        2π wraparound that hurts MSE.
        """
        # Forward axis in world → project to ground plane, normalize, take yaw
        # Convention: R's first column is the "forward" of the pose frame.
        fwd = R[:, 0]
        yaw = np.arctan2(fwd[1], fwd[0])
        return np.array([T[0], T[1], T[2], np.sin(yaw), np.cos(yaw)], np.float32)

    def _intrinsic_for_context(self, K_orig, orig_h, orig_w):
        """Scale a 3x3 intrinsic for the context resolution.

        Raw depth PNGs come at some native resolution. After resizing to
        (context_image_height, context_image_width), the intrinsic's fx, fy,
        cx, cy must be rescaled by the axis ratios.
        """
        sx = self.context_image_width / orig_w
        sy = self.context_image_height / orig_h
        K = K_orig.copy().astype(np.float32)
        K[0, 0] *= sx  # fx
        K[1, 1] *= sy  # fy
        K[0, 2] *= sx  # cx
        K[1, 2] *= sy  # cy
        return K

    def _unproject_depth(self, depth_hw1, K):
        """Unproject a (Hc, Wc, 1) depth image to (Hc, Wc, 3) camera-frame points.

        Invalid depths (==0) produce zero points.
        """
        H, W = depth_hw1.shape[:2]
        vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        uv1 = np.stack([us, vs, np.ones_like(us)], axis=-1).astype(np.float32)  # (H, W, 3)
        K_inv = np.linalg.inv(K).astype(np.float32)
        rays = uv1 @ K_inv.T  # (H, W, 3)
        d = depth_hw1[..., 0]
        valid = d > 0
        pts = rays * d[..., None]
        pts[~valid] = 0.0
        return pts.astype(np.float32)

    def _compute_geometry_gt(
        self,
        context_extrinsics,
        current_extrinsic,
        base_extrinsic,
        context_depths,
        context_mask,
        K_context,
    ):
        """Compute per-frame pose, local points, and world points relative to current.

        Args:
            context_extrinsics: (N, 4, 4) camera-to-world for each context frame.
                                Slots where mask=False are ignored (identity).
            current_extrinsic:  (4, 4) camera-to-world for the current (memory_start) frame.
            base_extrinsic:     (4, 4) chassis→camera calibration (constant per episode).
            context_depths:     (N, Hc, Wc, 1) resized depth (meters).
            context_mask:       (N,) bool, valid frames.
            K_context:          (3, 3) intrinsic matched to context resolution.

        Returns:
            gt_poses:         (N, 5)
            gt_local_points:  (N, Hc, Wc, 3)
            gt_world_points:  (N, Hc, Wc, 3) — in current-frame world
        """
        N = context_extrinsics.shape[0]
        Hc = self.context_image_height
        Wc = self.context_image_width

        R_cur = current_extrinsic[0:3, 0:3]
        T_cur = current_extrinsic[0:3, 3]

        gt_poses = np.zeros((N, 5), np.float32)
        gt_local = np.zeros((N, Hc, Wc, 3), np.float32)
        gt_world = np.zeros((N, Hc, Wc, 3), np.float32)

        for i in range(N):
            if not context_mask[i]:
                continue
            Ri = context_extrinsics[i, 0:3, 0:3]
            Ti = context_extrinsics[i, 0:3, 3]
            R_rel, T_rel = self.relative_pose(R_cur, T_cur, Ri, Ti, base_extrinsic)
            gt_poses[i] = self._pose_to_xyzsincos(R_rel, T_rel)

            local_pts = self._unproject_depth(context_depths[i], K_context)
            gt_local[i] = local_pts

            # world points (expressed in current chassis frame):
            #   p_world_cur = R_rel · p_cam + T_rel
            valid = local_pts.sum(axis=-1) != 0
            world_pts = local_pts @ R_rel.T + T_rel[None, None, :]
            world_pts[~valid] = 0.0
            gt_world[i] = world_pts.astype(np.float32)

        return gt_poses, gt_local, gt_world

    def _compute_subgoal_gt(self, point_goal):
        """Sub-pointgoal = final target in current frame. Placeholder; see file docstring."""
        return np.array(point_goal, np.float32)

    # ---- Main entry ----

    def __getitem__(self, index):
        start_time = time.time()

        (
            camera_intrinsic,
            trajectory_base_extrinsic,
            trajectory_extrinsics,
            trajectory_length,
        ) = self.process_data_parquet(index)

        trajectory_path_points, _ = self.process_path_points(index)
        trajectory_obstacle_points, _ = self.process_obstacle_points(index, trajectory_path_points)

        if self.prior_sample:
            pixel_start_choice, target_choice = self.rank_steps(
                trajectory_extrinsics, trajectory_obstacle_points
            )
            memory_start_choice = np.random.randint(pixel_start_choice, target_choice)
        else:
            pixel_start_choice = np.random.randint(0, trajectory_length // 2)
            target_choice = np.random.randint(pixel_start_choice + 1, trajectory_length - 1)
            memory_start_choice = np.random.randint(pixel_start_choice, target_choice)

        if self.random_digit:
            memory_digit = np.random.randint(2, 8)
            pred_digit = memory_digit
        else:
            memory_digit = 4
            pred_digit = 4

        memory_images, depth_image, _ = self.process_memory(
            self.trajectory_rgb_path[index],
            self.trajectory_depth_path[index],
            memory_start_choice,
            memory_digit=memory_digit,
        )

        context_rgb, context_depth, context_mask, context_indices = self.process_context_window(
            self.trajectory_rgb_path[index],
            self.trajectory_depth_path[index],
            memory_start_choice,
        )

        (
            target_local_points,
            augment_local_points,
            target_world_points,
            augment_world_points,
            action_indexes,
        ) = self.process_actions(
            trajectory_extrinsics,
            trajectory_base_extrinsic,
            memory_start_choice,
            target_choice,
            pred_digit=pred_digit,
        )

        init_vector = target_local_points[1] - target_local_points[0]
        target_xyt_actions = self.xyz_to_xyt(target_local_points, init_vector)
        augment_xyt_actions = self.xyz_to_xyt(augment_local_points, init_vector)
        pred_actions = target_xyt_actions[action_indexes]
        augment_actions = augment_xyt_actions[action_indexes]

        if trajectory_obstacle_points.shape[0] != 0:
            pred_distance = (
                np.abs(target_world_points[:, np.newaxis, 0:2] - trajectory_obstacle_points[np.newaxis, :, 0:2])
                .sum(axis=-1)
                .min(axis=-1)
            )
            augment_distance = (
                np.abs(augment_world_points[:, np.newaxis, 0:2] - trajectory_obstacle_points[np.newaxis, :, 0:2])
                .sum(axis=-1)
                .min(axis=-1)
            )
            pred_critic = (
                -5.0 * (pred_distance[action_indexes[:-1]] < 0.1).mean()
                + 0.5 * (pred_distance[action_indexes][1:] - pred_distance[action_indexes][:-1]).sum()
            )
            augment_critic = (
                -5.0 * (augment_distance[action_indexes[:-1]] < 0.1).mean()
                + 0.5 * (augment_distance[action_indexes][1:] - augment_distance[action_indexes][:-1]).sum()
            )
        else:
            pred_critic = 2.0
            augment_critic = 2.0

        point_goal = target_xyt_actions[-1]

        # --- geometry GT (new) ---
        # Use context_indices clipped into valid range; invalid slots already masked.
        clipped_ctx = np.clip(context_indices, 0, trajectory_length - 1)
        ctx_ext = trajectory_extrinsics[clipped_ctx]  # (N, 4, 4)
        cur_ext = trajectory_extrinsics[memory_start_choice]

        # original depth size → we need the native H,W to scale K correctly.
        # Read one depth directly to know its size.
        sample_depth = np.array(Image.open(self.trajectory_depth_path[index][0]), np.uint16)
        orig_h, orig_w = sample_depth.shape[:2]
        K_context = self._intrinsic_for_context(camera_intrinsic, orig_h, orig_w)

        gt_poses, gt_local_points, gt_world_points = self._compute_geometry_gt(
            ctx_ext,
            cur_ext,
            trajectory_base_extrinsic,
            context_depth,
            context_mask,
            K_context,
        )

        gt_subgoal = self._compute_subgoal_gt(point_goal)

        # action deltas (same as NavDP)
        pred_actions = (pred_actions[1:] - pred_actions[:-1]) * 4.0
        augment_actions = (augment_actions[1:] - augment_actions[:-1]) * 4.0

        # timing log (inherited pattern)
        end_time = time.time()
        self.item_cnt += 1
        self.batch_time_sum += end_time - start_time
        if self.item_cnt % self.batch_size == 0:
            avg_time = self.batch_time_sum / self.batch_size
            print(
                f'__getitem__ pid={os.getpid()}, avg_time(last {self.batch_size})={avg_time:.2f}s, cnt={self.item_cnt}'
            )
            self.batch_time_sum = 0.0

        return {
            'point_goal': torch.tensor(point_goal, dtype=torch.float32),
            'memory_rgb': torch.tensor(memory_images, dtype=torch.float32),
            'memory_depth': torch.tensor(depth_image, dtype=torch.float32),
            'context_rgb': torch.tensor(context_rgb, dtype=torch.float32),
            'context_depth': torch.tensor(context_depth, dtype=torch.float32),
            'pred_actions': torch.tensor(pred_actions, dtype=torch.float32),
            'augment_actions': torch.tensor(augment_actions, dtype=torch.float32),
            'pred_critic': torch.tensor(pred_critic, dtype=torch.float32),
            'augment_critic': torch.tensor(augment_critic, dtype=torch.float32),
            'gt_camera_poses': torch.tensor(gt_poses, dtype=torch.float32),
            'gt_local_points': torch.tensor(gt_local_points, dtype=torch.float32),
            'gt_world_points': torch.tensor(gt_world_points, dtype=torch.float32),
            'gt_subgoal': torch.tensor(gt_subgoal, dtype=torch.float32),
        }


def logoplanner_collate_fn(batch):
    return {
        'batch_pg':              torch.stack([b['point_goal']       for b in batch]),
        'batch_memory_rgb':      torch.stack([b['memory_rgb']       for b in batch]),
        'batch_memory_depth':    torch.stack([b['memory_depth']     for b in batch]),
        'batch_context_rgb':     torch.stack([b['context_rgb']      for b in batch]),
        'batch_context_depth':   torch.stack([b['context_depth']    for b in batch]),
        'batch_labels':          torch.stack([b['pred_actions']     for b in batch]),
        'batch_augments':        torch.stack([b['augment_actions']  for b in batch]),
        'batch_label_critic':    torch.stack([b['pred_critic']      for b in batch]),
        'batch_augment_critic':  torch.stack([b['augment_critic']   for b in batch]),
        'batch_gt_camera_poses': torch.stack([b['gt_camera_poses']  for b in batch]),
        'batch_gt_local_points': torch.stack([b['gt_local_points']  for b in batch]),
        'batch_gt_world_points': torch.stack([b['gt_world_points']  for b in batch]),
        'batch_gt_subgoal':      torch.stack([b['gt_subgoal']       for b in batch]),
    }

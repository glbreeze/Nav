# LoGoPlanner Training Notes

Setup, dataset preparation, and trainer design notes for reproducing **Stage 2** of LoGoPlanner training on top of the NavDP codebase, using the InternData-N1 mini dataset.

---

## 1. Dataset

### 1.1 Download

Mini split (v0.1):

```bash
hf download InternRobotics/InternData-N1 \
  --repo-type dataset \
  --revision v0.1-mini \
  --local-dir /scratch/lg154/Research/Nav/InternData-N1 \
  --max-workers 8
```

### 1.2 Loader patch

`InternNav/internnav/dataset/navdp_dataset_lerobot.py:177`

```diff
- camera_trajectory = np.array([np.stack(frame) for frame in df['action']], dtype=np.float64)
+ camera_trajectory = np.array([np.stack(frame) for frame in df['action']], dtype=np.float64).reshape(-1, 4, 4)
```

---

## 2. Running Training

Edit `InternNav/scripts/train/configs/navdp.py`:

```python
root_dir       = '/home/asus/Research/datasets/InternData-N1/vln_n1/traj_data_navdp'
dataset_navdp  = '/tmp/navdp_cache/apartment_1.json'
batch_size     = 2     # debug
num_workers    = 0     # debug
```

Drop a `breakpoint()` at `navdp_trainer.py:80` (start of `compute_loss`), then:

```bash
cd /home/asus/Research/InternNav
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
MASTER_ADDR=localhost MASTER_PORT=12345 \
python scripts/train/train.py --name navdp_debug --model-name navdp
```

---

## 3. Ground-Truth Availability

All GT signals required by the paper are present in the dataset:

| Signal | Source |
|---|---|
| Per-frame 4×4 camera pose | `parquet['action']` (loader exposes as `camera_trajectory`) |
| Per-frame intrinsic | `parquet['observation.camera_intrinsic']` |
| Per-frame depth | uint16 PNG → unproject with intrinsic for GT local points |
| GT world points | local points × extrinsic |
| Chassis-to-camera extrinsic `T_ext` | Fixed per episode (extrinsic from parquet row 0) |

---

## 4. Paper-Specified Training Schedule

From Sec. V.A and IV.B:

| Stage | Duration | Batch | Trainable | Frozen |
|---|---|---|---|---|
| 1 | 24 h | 12 | Geometry decoder + `camera_pose_head`, `local_point_head`, `world_point_head` | ViT encoder |
| 2 | 3 days | 32 | Diffusion head + task-specific heads | Geometry backbone decoder |

> **Scope of this work:** only Stage 2 is replicated.

---

## 5. Loss Terms (Paper Eqs. 2, 4, 6, 11)

1. **Local points** (Eq. 2): $L_{\text{local}} = \|\hat{P}_{\text{local}} - P_{\text{local}}^{\text{gt}}\|$, where $P_{\text{local}}^{\text{gt}} = D(u, v) \cdot K^{-1} \cdot [u, v, 1]^T$
2. **Camera pose** (Eq. 4): $L_{\text{pose}} = \|\hat{T}_c - T_c^{\text{gt}}\|$, parametrized as $(x, y, \theta)$ on the ground plane (3 DoF; the code's head outputs 5-dim, so we need a `[x, y, z, sin θ, cos θ]` decoding).
3. **World points** (Eq. 6): $L_{\text{world}} = \|\hat{P}_{\text{world}} - P_{\text{world}}^{\text{gt}}\|$ with sign-preserving exp parametrization.
4. **Diffusion** (Eq. 11): standard DDPM ε-prediction on $a_t = (\Delta x_t, \Delta y_t, \Delta\theta_t)$, $T = 24$.
5. **Goal / sub-goal**: appears in Table III ablation; loss form not specified (`pg_pred_mlp` head).
6. **Critic**: not in the paper, but `critic_head` and `cs_pred_mlp` exist in the checkpoint — likely retained from NavDP training.

### 5.1 Unknowns from the paper

- Loss weights (`λ_local`, `λ_pose`, `λ_world`, `λ_diffusion`, `λ_goal`, `λ_critic`)
- Norm type per term (L1 / L2 / Huber)
- LR, optimizer, schedule

### 5.2 Defaults chosen here

| Term | Weight | Rationale |
|---|---|---|
| diffusion | 1.0 | Matches NavDP default |
| critic | 1.0 | Matches NavDP default |
| pose | 1.0 | Most important ablation contribution (Table III) |
| local | 0.5 | Dense per-pixel — dampened |
| world | 0.5 | Dense per-pixel — dampened |
| subgoal | 0.1 | Small, only 3-dim regression |

MSE everywhere — the paper doesn't specify norms and NavDP uses `.square().mean()` throughout.

---

## 6. Trainer Design

Mirrors `NavDPTrainer`'s structure: same `__init__`, optimizer, scheduler, dataloader, and `save_model` patterns. Only `compute_loss` diverges.

### 6.1 Loss composition

| Term | Source | Default weight |
|---|---|---|
| action (diffusion ε-pred, ng + mg) | NavDP style, Eq. 11 | 1.0 |
| critic | NavDP (`critic_head`) | 1.0 |
| pose | Eq. 4, `ExtrinctHead` output | 1.0 |
| local | Eq. 2, local-point head | 0.5 |
| world | Eq. 6, world-point head | 0.5 |
| subgoal | Table III "Goal" col, `pg_pred_mlp` | 0.1 |

### 6.2 Two-stage training is config-driven, not trainer-driven

- **Stage 1 config:** `w_diffusion = 0`, `w_critic = 0`, `w_subgoal = 0`; unfreeze geometry in model.
- **Stage 2 config:** all weights active; freeze geometry in model.

### 6.3 Contracts

- **Forward output (dict):** `noise_pred_ng`, `noise_pred_mg`, `ng_noise`, `mg_noise`, `label_critic_pred`, `augment_critic_pred`, `camera_poses_pred`, `local_points_pred`, `world_points_pred`, `subgoal_pred`.
- **Batch (13 keys):** documented in the trainer docstring; this is the next piece to build (dataset + collate).

### 6.4 Open decisions (flagged for review)

1. **Camera-pose GT encoding** — `ExtrinctHead.fc_pose` outputs 5-dim, but the paper specifies 3 DoF $(x, y, \theta)$. Pose GT is currently `[B, N, P]` with `P` TBD. Most likely `[x, y, z, sin θ, cos θ]`; alternative is `[x, y, sin θ, cos θ, scale]`. Will be finalized when writing the model.
2. **Sub-goal GT** — paper mentions "Goal" supervision but doesn't specify the target. Best guess: final waypoint of the trajectory, expressed in the current frame.
3. **Critic** — not in the paper; kept the NavDP-style critic loss. Set `w_critic = 0` to disable.
4. **MSE everywhere** — no norm specified by the paper; NavDP convention used.

---

## 7. Stage 2 — Paper Concept ↔ Checkpoint Mapping

| Paper concept | Checkpoint keys | Stage-2 treatment |
|---|---|---|
| Pi3 image encoder (DINOv2) | `state_encoder.encoder` (343) | Always frozen via `no_grad` in `forward_image` |
| Decoder of the video geometry model | `state_encoder.{decoder, camera_decoder, point_decoder}` (648 + 64 + 64) | Freeze, load from ckpt ✓ |
| Task-specific heads (scene + extrinsics) | `state_encoder.{world_point_decoder, world_point_head, wp_head, camera_head, point_head, fusion_head}` | Load from ckpt, keep training ✓ |
| State / scene tokenizers (diffusion conditioning) | `state_encoder.{former_net, former_pe, former_query, state_layer, state_compressor, scene_layer, scene_compressor}` | Load from ckpt, keep training |
| Depth scale prior | `state_encoder.depth_model` (DA-V2) | Load from ckpt, keep training (paper: "depth-based scale priors are injected") |

---

## 8. Diff: `NavDPTrainer` vs `LoGoPlannerTrainer`

### 8.1 Loss function (the main difference)

**NavDPTrainer — 3 terms**

- `action_loss`: $0.5 \cdot \text{ng} + 0.5 \cdot \text{mg}$ (diffusion noise MSE)
- `critic_loss`: label + augment
- `aux_loss`: $0.5 \cdot (pg - \text{aux\_pred}[0])^2 + 0.5 \cdot (pg - \text{aux\_pred}[1])^2$ (point-goal aux prediction)

Total:

```text
0.8 · action_loss + 0.2 · critic_loss + 0.5 · aux_loss
```

**LoGoPlannerTrainer — 6 terms** (paper Sec. IV.B)

- `action_loss` — same as NavDP
- `critic_loss` — same as NavDP
- `pose_loss` — camera extrinsic regression (Eq. 4)
- `local_loss` — local 3D points: $D \cdot K^{-1} \cdot [u, v, 1]$
- `world_loss` — world 3D points: $T_{cw} \cdot \text{local}$
- `subgoal_loss` — sub-pointgoal MLP

Weights pulled from `config.il.loss` (configurable; defaults: diffusion 1.0, critic 1.0, pose 1.0, local 0.5, world 0.5, subgoal 0.1).

### 8.2 Model inputs

- **NavDP:** `batch_pg`, `batch_ig`, `batch_tg`, `batch_rgb`, `batch_depth`, `batch_labels`, `batch_augments`
- **LoGoPlanner:** drops `batch_ig` / `batch_tg` / `batch_rgb` / `batch_depth`; adds memory + context streams + GT geometry:
  - `batch_memory_rgb`, `batch_memory_depth` (last-frame depth)
  - `batch_context_rgb`, `batch_context_depth` (N = 12 context frames)
  - `batch_gt_camera_poses`, `batch_gt_local_points`, `batch_gt_world_points`, `batch_gt_subgoal`

---

## 9. Checkpoint Structure (2242 keys, top-level under `LoGoPlanner_Policy`)

- **Geometry stack (Pi3-based):** `state_encoder.encoder` (343); `state_encoder.{decoder, camera_decoder, point_decoder, world_point_decoder, conf_decoder}` (~648 + 64 × 4)
- **Geometry heads:** `state_encoder.{camera_head, point_head, world_point_head, fusion_head, wp_head, conf_head}`
- **State / scene tokenizers:** `state_encoder.{former_net, former_pe, former_query, state_layer, state_compressor, scene_layer, scene_compressor}`
- **Depth-V2 priors:** `state_encoder.depth_model`, `rgbd_encoder.depth_model`, `rgbd_encoder.rgb_model`
- **RGBD encoder:** `rgbd_encoder.{former_net, former_query, former_pe, project_layer}`
- **Diffusion stack:** `decoder`, `decoder_layer`, `input_embed`, `action_head`, `critic_head`, `cond_pos_embed`, `out_pos_embed`, `layernorm`, `pg_pred_mlp`, `start_encoder`, `state_decoder`, `point_encoder` *(unused)*, `cs_pred_mlp` *(unused)*
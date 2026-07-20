"""MemNav eval agent — streaming LingBot cache + InternNav MemNavNet inference."""

import os
import sys

import cv2
import numpy as np
import torch
from matplotlib import colormaps as cm

_INTERNAV = os.environ.get(
    "INTERNNAV_REPO",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../InternNav")),
)
if _INTERNAV not in sys.path:
    sys.path.insert(0, _INTERNAV)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_INTERNAV, "src", "diffusion-policy"))

from internnav.model.basemodel.memnav.memnav_policy import MemNavModelConfig, MemNavPolicy  # noqa: E402
from scripts.train.configs.memnav import memnav_exp_cfg  # noqa: E402
from stream_state import LingBotEpisodeState  # noqa: E402


def _strip_prefix(state):
    if not isinstance(state, dict):
        return state
    if any(k.startswith("core.") for k in state):
        return state
    if any(k.startswith("policy.") for k in state):
        return {(k[len("policy."):] if k.startswith("policy.") else k): v for k, v in state.items()}
    return {f"core.{k}": v for k, v in state.items()}


class MemNav_Agent:
    def __init__(
        self,
        image_intrinsic,
        image_size=518,
        memory_size=8,
        predict_size=24,
        temporal_depth=8,
        heads=8,
        token_dim=384,
        navi_model="./memnav.ckpt",
        device="cuda:0",
        lingbot_repo=None,
        lingbot_weights=None,
    ):
        self.image_intrinsic = image_intrinsic
        self.device = device
        self.predict_size = predict_size
        self.image_size = image_size
        self.memory_size = memory_size
        self.window = memory_size

        # The professor's config has no stage1/stage2 split (that was a peer-side addition), so
        # the single memnav_exp_cfg is the source of truth. It reads window/num_scale/
        # max_frame_num from the env, so exporting MEMNAV_WINDOW here lines the model and the
        # stream up with however the caches were precomputed. train_stage below is a no-op on
        # this code (IlCfg is extra='allow'); kept so the agent works against either checkout.
        cfg = memnav_exp_cfg.model_dump()
        if lingbot_repo:
            cfg["il"]["lingbot_repo"] = lingbot_repo
        if lingbot_weights:
            cfg["il"]["lingbot_weights"] = lingbot_weights
        cfg["il"]["train_stage"] = 2
        model_cfg = MemNavModelConfig(model_cfg=cfg)
        self.policy = MemNavPolicy.from_pretrained(navi_model, config=model_cfg)
        self.policy.to(device)
        self.policy.eval()
        self.core = self.policy.core
        self.core.train_stage = 2
        if self.core.noise_scheduler is None:
            self.core._init_noise_scheduler(self.core.noise_scheduler_iters)

        ckpt = torch.load(navi_model, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and not any(k.startswith("core.") for k in ckpt):
            ckpt = _strip_prefix(ckpt)
        inc = self.policy.load_state_dict(ckpt, strict=False)
        print(f"[MemNav_Agent] loaded {navi_model}: missing={len(inc.missing_keys)} "
              f"unexpected={len(inc.unexpected_keys)}", flush=True)

        self.target_H, self.target_W = 168, 308
        self.stream_states = []
        self.goal_tensors = []
        # Revisit candidate region E(k) = [anchor_margin .. k - exclude_recent], identical to
        # what the loader labels at train time. Keep the default in sync with
        # MemNav_Dataset.exclude_recent (83 = p90 of the covis-onset distance); a mismatch here
        # means inference scores candidates the model was never trained to retrieve.
        self.anchor_margin = self.core.num_scale + self.window - 1
        self.exclude_recent = int(os.environ.get("MEMNAV_EXCLUDE_RECENT", "83"))
        print(f"[MemNav_Agent] window={self.window} E(k)=[{self.anchor_margin}..k-{self.exclude_recent}]",
              flush=True)

    def reset(self, batch_size, threshold):
        print("================ MemNav Agent Reset ================", flush=True)
        self.batch_size = int(batch_size)
        self.stop_threshold = threshold
        self.stream_states = [
            LingBotEpisodeState(self.core.lingbot, device=self.device)
            for _ in range(self.batch_size)
        ]
        self.goal_tensors = [None] * self.batch_size
        self.goal_cls = [None] * self.batch_size
        # Last retrieval readout per env, exposed over HTTP for the revisit eval. SR alone
        # cannot tell "the memory found the place but the policy fumbled it" apart from "the
        # memory never found it": these two are the direct measurements, SR is downstream of
        # MPC, collision sliding and trajectory choice. None until the stream warms up.
        self._last_gate = [None] * self.batch_size
        self._last_match = [None] * self.batch_size
        # {m: goal_pose} per env, plus a "_dirty" marker encode_memory sets when it had to
        # recompute (and therefore clobbered the live KV cache). Episode-scoped: cleared in
        # reset_env, and encode_memory keeps only the current anchor.
        self._goal_pose_cache = [{} for _ in range(self.batch_size)]

    def reset_env(self, i):
        self.stream_states[i].reset()
        self._goal_pose_cache[i] = {}
        self.goal_tensors[i] = None
        self.goal_cls[i] = None

    def process_image(self, images):
        out = []
        for img in images:
            resize = cv2.resize(img, (self.target_W, self.target_H))
            out.append(resize.astype(np.float32) / 255.0)
        return np.array(out)

    def _lingbot_frame(self, rgb_bgr):
        """NavDP BGR uint8/float HxWx3 -> LingBot [3,518,518] tensor."""
        rgb = cv2.cvtColor((rgb_bgr * 255).astype(np.uint8) if rgb_bgr.max() <= 1.0 else rgb_bgr,
                           cv2.COLOR_BGR2RGB)
        tmp = os.path.join("/tmp", f"memnav_{os.getpid()}_{id(rgb)}.jpg")
        cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        t = self.core.lingbot.load_images([tmp])[0]
        try:
            os.remove(tmp)
        except OSError:
            pass
        return t

    def _goal_tensor(self, goal_bgr, env_i):
        if self.goal_tensors[env_i] is None:
            g = self._lingbot_frame(goal_bgr)
            self.goal_tensors[env_i] = g
            self.goal_cls[env_i] = self.core.lingbot.dino(g.unsqueeze(0))["cls"][0]
        return self.goal_tensors[env_i]

    def _warmup_traj(self, images):
        bs = images.shape[0]
        t = np.zeros((bs, 1, self.predict_size, 3), dtype=np.float32)
        t[:, :, :, 1] = 0.1
        v = np.ones((bs, 1), dtype=np.float32)
        mask = self.project_trajectory(images, t, v)
        return t[:, 0], t, v, mask

    def project_trajectory(self, images, n_trajectories, n_values):
        trajectory_masks = []
        img_h = images[0].shape[0]
        for i in range(images.shape[0]):
            trajectory_mask = np.array(images[i], copy=True)
            n_trajectory = n_trajectories[i, :, :, 0:2]
            n_value = n_values[i]
            for waypoints, value in zip(n_trajectory, n_value):
                norm_value = np.clip(-float(value) * 0.1, 0, 1)
                color = np.array(cm.get("jet")(norm_value)[0:3]) * 255.0
                input_points = np.zeros((waypoints.shape[0], 3), dtype=np.float32) - 0.2
                input_points[:, 0:2] = waypoints
                input_points[:, 1] = -input_points[:, 1]
                camera_z = (
                    img_h - 1
                    - self.image_intrinsic[1][1] * input_points[:, 2] / (input_points[:, 0] + 1e-8)
                    - self.image_intrinsic[1][2]
                )
                camera_x = (
                    self.image_intrinsic[0][0] * input_points[:, 1] / (input_points[:, 0] + 1e-8)
                    + self.image_intrinsic[0][2]
                )
                for j in range(camera_x.shape[0] - 1):
                    try:
                        if (
                            camera_x[j] > 0 and camera_z[j] > 0
                            and camera_x[j + 1] > 0 and camera_z[j + 1] > 0
                        ):
                            p1 = (int(camera_x[j]), int(camera_z[j]))
                            p2 = (int(camera_x[j + 1]), int(camera_z[j + 1]))
                            trajectory_mask = cv2.line(
                                trajectory_mask, p1, p2, color.astype(np.uint8).tolist(), 5)
                    except Exception:
                        pass
            trajectory_masks.append(trajectory_mask)
        return np.concatenate(trajectory_masks, axis=1)

    def stream_frame(self, images):
        """Append rendered frames to memory WITHOUT running the policy.

        The revisit eval drives the robot along a GT path first (leg 1: walk past the future
        goal; leg 2: walk away), so that the goal is genuinely in memory -- and far enough back
        to land inside E(k) -- before the policy is asked to navigate back to it. step_imagegoal
        cannot do this job: it requires a goal image, and _goal_tensor caches goal_cls on the
        first call, so a placeholder passed during leg 1 would be frozen in and the real goal
        ignored. reset_env would clear the goal but also wipe the stream state we are building.
        Returns the current step index k.
        """
        # RAW frame, not process_image's 308x168 -- see step_imagegoal for why.
        for i in range(self.batch_size):
            frame = self._lingbot_frame(images[i])
            self.stream_states[i].append_frame(frame)
        return len(self.stream_states[0].frames) - 1

    def set_goal(self, goals):
        """Install the goal image (and its CLS) after the memory has been built, replacing any
        earlier one. Separate from reset_env, which would also wipe the memory."""
        for i in range(self.batch_size):
            # same path as step_imagegoal's goal: straight to _lingbot_frame, no
            # process_image (that one resizes to the NavDP 168x308 and normalizes; goals
            # go to LingBot's 518 square directly, as they do at train time).
            g = self._lingbot_frame(goals[i])
            self.goal_tensors[i] = g
            self.goal_cls[i] = self.core.lingbot.dino(g.unsqueeze(0))["cls"][0]

    def step_imagegoal(self, goals, images, depths):
        batch = {
            "batch_goal_cls": [],
            "batch_mem_cls": [],
            "batch_mem_mask": [],
            "batch_cand_mask": [],
            "batch_goal_image": [],
            "batch_window_images": [],
            "batch_online_caches": [],
            "batch_match_window_images": [],
            "batch_goal_warm_frames": [],
            "batch_goal_pose_cache": [],
            "cur_steps": [],
        }
        ready = True
        for i in range(self.batch_size):
            # RAW frame -- NOT proc. process_image resizes to NavDP's 308x168, so feeding
            # it here put every stored frame through a 10x downsample and back up to 518,
            # while the goal image reached _lingbot_frame untouched. Measured on 24 real
            # frames (job 14185522): cos(goal_path_i, mem_path_i) = 0.85, which is LOWER
            # than cos between two genuinely different scenes on one path (0.91) -- so
            # retrieval scored 8/24 with the paths split and 24/24 with them matched.
            # That is gate=0.066 / match_hit=0.0 in the revisit eval. Training compares
            # full-res jpgs on both sides, so this existed only at inference.
            frame = self._lingbot_frame(images[i])
            st = self.stream_states[i]
            st.append_frame(frame)
            # goals arrives from the server as uint8 0-255 and does NOT go through
            # process_image (only `images` does), so it is already in _lingbot_frame's
            # expected range. The old `* 255` here inverted it: uint8 arithmetic wraps and
            # 255 == -1 (mod 256), so x*255 -> 256-x, i.e. a photographic negative. Structure
            # survived (hence eval never scored 0) but every goal the policy saw at eval was
            # a negative of what it was trained on. _lingbot_frame scales only when max<=1.
            goal_rgb = self._goal_tensor(goals[i], i)
            k = len(st.frames) - 1
            # window_forward recomputes [k-W+1..k]; the clean lingbot-map has no %len rope
            # wrap, so k must be >= anchor_margin for a fully populated window -- the same
            # min-k the loader enforces at train time. Below it, warm up (drive straight).
            if not st.ready() or k < self.anchor_margin:
                ready = False
                continue
            mem = st.mem_cls_tensor()
            win = st.window_tensor(k, self.window)
            cache = st.build_cache()
            # E(k) = [anchor_margin .. k - exclude_recent] — the same candidate region the loader
            # labels at train time. Scoring ALL history (the old ones(L) mask) would let the
            # current-approach frames compete, and those have the highest cosine of all: the
            # robot sees the goal because it is walking at it. Training routes exactly those to
            # the novel branch, so retrieving them here is the one answer training ruled out.
            # Empty region (k < anchor_margin + exclude_recent, i.e. memory still too short for
            # any loop closure) -> the head floors max_cos to -1 -> gate ~= 0 -> novel, correct.
            cand = torch.zeros(1, mem.shape[0], dtype=torch.bool, device=self.device)
            hi = k - self.exclude_recent
            if hi >= self.anchor_margin:
                cand[0, self.anchor_margin:hi + 1] = True
            match_idx, gate_logit, _ret_logits = self.core.retrieval(
                self.goal_cls[i].unsqueeze(0).to(self.device),
                mem.unsqueeze(0).to(self.device),
                cand,
            )
            # min(anchor_margin,k-1) low bound keeps m<=k-1 (frames[m] exists) in early online steps
            m = int(match_idx[0].clamp(min(self.anchor_margin, k - 1), k - 1).item())
            self._last_gate[i] = float(torch.sigmoid(gate_logit[0]).item())
            self._last_match[i] = m
            batch["batch_goal_cls"].append(self.goal_cls[i])
            batch["batch_mem_cls"].append(mem)
            batch["batch_mem_mask"].append(torch.ones(mem.shape[0], dtype=torch.bool))
            batch["batch_cand_mask"].append(cand[0].cpu())
            batch["batch_goal_image"].append(goal_rgb)
            batch["batch_window_images"].append(win)
            batch["batch_online_caches"].append(cache)
            batch["batch_match_window_images"].append(st.match_window_tensor(m, self.window))
            # goal warm-up frames [max(num_scale, m-goal_warm+1) .. m], from memory not disk.
            # Same m the encode_memory anchor will clamp to (same retrieval, same inputs), so the
            # range lines up with goal_append_warm_frames' internal start.
            batch["batch_goal_warm_frames"].append(st.warm_frames(m, self.core.goal_warm))
            batch["batch_goal_pose_cache"].append(self._goal_pose_cache[i])
            batch["cur_steps"].append(k)

        if not ready or len(batch["cur_steps"]) == 0:
            return self._warmup_traj(images)

        B = len(batch["cur_steps"])
        D = batch["batch_mem_cls"][0].shape[-1]
        Lmax = max(x.shape[0] for x in batch["batch_mem_cls"])
        mem_cls = torch.zeros(B, Lmax, D)
        mem_mask = torch.zeros(B, Lmax, dtype=torch.bool)
        cand_mask = torch.zeros(B, Lmax, dtype=torch.bool)
        for bi, m in enumerate(batch["batch_mem_cls"]):
            mem_cls[bi, : m.shape[0]] = m
            mem_mask[bi, : m.shape[0]] = True
        for bi, c in enumerate(batch["batch_cand_mask"]):
            cand_mask[bi, : c.shape[0]] = c
        model_batch = {
            "batch_goal_cls": torch.stack(batch["batch_goal_cls"]),
            "batch_mem_cls": mem_cls,
            "batch_mem_mask": mem_mask,
            "batch_cand_mask": cand_mask,
            "batch_goal_image": torch.stack(batch["batch_goal_image"]),
            "batch_window_images": torch.stack(batch["batch_window_images"]),
            "batch_online_caches": batch["batch_online_caches"],
            "batch_match_window_images": torch.stack(batch["batch_match_window_images"]),
            # kept as a list, NOT stacked: warm-frame count varies per sample (m-start+1, and
            # start=max(num_scale, m-warm+1) clips when m is small). encode_memory indexes [b].
            "batch_goal_warm_frames": batch["batch_goal_warm_frames"],
            # per-env {m: goal_pose} dicts. Must be forwarded explicitly: model_batch is a
            # fresh dict, so anything left in `batch` is invisible to encode_memory.
            "batch_goal_pose_cache": batch["batch_goal_pose_cache"],
            "cur_steps": batch["cur_steps"],
        }
        # Snapshot only the envs whose goal_pose is about to be recomputed -- those are the
        # only ones goal_append_warm will clobber. m is stable within an episode, so in steady
        # state this dict is empty and nothing is cloned.
        _kv_snaps = {}
        for _i in range(self.batch_size):
            if self._last_match[_i] is not None and self._last_match[_i] not in self._goal_pose_cache[_i]:
                _kv_snaps[_i] = self.stream_states[_i].snapshot_kv()
        all_traj, all_vals, good_traj, bad_traj = self.core.predict_action(model_batch, sample_num=int(__import__("os").environ.get("MEMNAV_SAMPLE_NUM","8")))
        # Put back the live KV cache wherever the goal had to be recomputed: that call wiped
        # it (goal_append_warm -> _inject -> clean_kv_cache) and the next append_frame would
        # otherwise extend the goal warm-up context instead of the stream.
        for _i, _snap in _kv_snaps.items():
            self._goal_pose_cache[_i].pop("_dirty", None)
            self.stream_states[_i].restore_kv(_snap)
        if all_vals.max() < self.stop_threshold:
            good_traj[:, :, :, 0] = 0.0
            good_traj[:, :, :, 1] = np.sign(good_traj[:, :, :, 1].mean())
        mask = self.project_trajectory(images, all_traj, all_vals)
        return good_traj[:, 0], all_traj, all_vals, mask

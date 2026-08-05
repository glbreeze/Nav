#!/usr/bin/env python
"""Is the goal's pose -- and therefore the direction the policy is told to go -- correct?

Retrieval is now provably right (match=117 against a ground-truth anchor of 120), yet the
policy drives at +65 deg while the goal sits at -138 deg. Everything between "found the frame"
and "produced an action" runs through one geometric quantity:

    goal_pose  = camera_pose(goal_append_warm(goal_img, cache, m, ...))   <- NEVER VERIFIED
    cur_pose   = cache["cam_pose_enc"][k]                                 <- verified (0.0078)
    t_rel      = R_cur^T (t_goal - t_cur)

The earlier smoke test only checked cur_pose against the precomputed disk values. goal_pose
comes from a different path -- inject history, warm 64 frames, stream the goal at m+1 -- and
has never been compared to anything.

Decisive test: feed the frame at index m ITSELF as the goal image. Then goal_pose must come
back equal to that frame's own recorded pose. Any large gap localises the bug to the goal-pose
path; a small gap clears it and points at the model being undertrained instead.

Also reports the bearing the policy would be given vs the true bearing, since a pose that is
"close" in metres can still be wrong in the direction that matters.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get(
    "MEMNAV_SERVER_DIR", "/scratch/ay2710/Nav_memnav/NavDP/baselines/memnav"))
from stream_state import LingBotEpisodeState  # noqa: E402

from internnav.model.basemodel.memnav.memnav_policy import MemNavModelConfig, MemNavPolicy  # noqa: E402
from scripts.train.configs.memnav import memnav_exp_cfg  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root_dir", default="/scratch/ay2710/mp3d_data/vln_n1/traj_data")
    ap.add_argument("--feature_root", default="/scratch/ay2710/mp3d_feat/vln_n1/traj_data")
    ap.add_argument("--n_frames", type=int, default=260)
    args = ap.parse_args()
    dev = "cuda"

    cfg = memnav_exp_cfg.model_dump()
    cfg["il"]["root_dir"] = args.root_dir
    cfg["il"]["feature_root"] = args.feature_root
    for e, c in (("LINGBOT_REPO", "lingbot_repo"), ("LINGBOT_WEIGHTS", "lingbot_weights")):
        if os.environ.get(e):
            cfg["il"][c] = os.environ[e]
    assert cfg["il"].get("lingbot_weights"), "LINGBOT_WEIGHTS not set"

    policy = MemNavPolicy(MemNavModelConfig(model_cfg=cfg)).to(dev).eval()
    core = policy.core
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    policy.load_state_dict(sd if isinstance(sd, dict) else sd.state_dict(), strict=False)

    RGB_SUB = "observation.images.rgb"
    rgb_dir = cam = None
    for p in sorted(glob.glob(
            f"{args.feature_root}/mp3d_2leg/*/*/videos/chunk-000/lingbot_cam_cache.npz")):
        rel = os.path.relpath(p, args.feature_root)
        rd = os.path.join(args.root_dir, os.path.dirname(rel), RGB_SUB)
        if len(glob.glob(f"{rd}/*.jpg")) >= args.n_frames:
            rgb_dir, cam = rd, p
            break
    assert rgb_dir is not None
    disk = np.load(cam)["cam_pose_enc"]
    print(f"[traj] {rgb_dir}\n[disk] cam_pose_enc {disk.shape}\n", flush=True)

    N = args.n_frames
    imgs = core.lingbot.load_images([os.path.join(rgb_dir, f"{i}.jpg") for i in range(N)])
    st = LingBotEpisodeState(core.lingbot, device=dev)
    for j in range(N):
        st.append_frame(imgs[j].to(dev))
    cache = st.build_cache()
    k = N - 1
    am = core.num_scale + core.window - 1
    excl = int(os.environ.get("MEMNAV_EXCLUDE_RECENT", "83"))

    mem = st.mem_cls_tensor()
    print(f"{'':>5} {'---- ONLINE (eval) ----':>17} {'':>14} {'':>13} {'':>8} "
          f"| {'-- DISK (train) --':>9} {'':>8}", flush=True)
    print(f"{'m':>5} {'|pose err| m':>17} {'model brg':>14} {'true brg':>13} "
          f"{'err deg':>8} | {'|err| m':>9} {'err deg':>8}", flush=True)
    print("-" * 88, flush=True)

    rows = []
    for m in range(am, k - excl + 1, max(1, (k - excl - am) // 6)):
        goal_img = imgs[m].to(dev)                      # the goal IS frame m
        goal_cls = core.lingbot.dino(goal_img.unsqueeze(0))["cls"][0]
        cand = torch.zeros(1, mem.shape[0], dtype=torch.bool, device=dev)
        cand[0, m] = True                               # force the anchor to m
        batch = {
            "batch_goal_cls": goal_cls.unsqueeze(0),
            "batch_mem_cls": mem.unsqueeze(0).to(dev),
            "batch_mem_mask": torch.ones(1, mem.shape[0], dtype=torch.bool),
            "batch_cand_mask": cand,
            "batch_goal_image": goal_img.unsqueeze(0),
            "batch_window_images": st.window_tensor(k, core.window).unsqueeze(0).to(dev),
            "batch_online_caches": [cache],
            "batch_goal_warm_frames": [st.warm_frames(m, core.goal_warm)],
            "cur_steps": [k],
        }
        enc = core.encode_memory(batch)
        gp = enc["goal_pose"][0].float().cpu().numpy()
        cp = enc["cur_pose"][0].float().cpu().numpy()

        # SAME m through the DISK path -- the one training uses. If this is also wrong the
        # checkpoint was fit against broken goal geometry; if it is right, the online cache
        # I added is the culprit and training is unaffected.
        dbatch = dict(batch)
        dbatch.pop("batch_online_caches")
        dbatch.pop("batch_goal_warm_frames")
        dbatch["cache_paths"] = [cam.replace("lingbot_cam_cache.npz", "lingbot_cache.npz")]
        # _load_cache falls back to get_scale_kv(rgb_dir) when the npz was written with
        # --skip_scale, and goal_append_warm reads its warm-up jpgs from here too.
        dbatch["rgb_dirs"] = [rgb_dir]
        try:
            denc = core.encode_memory(dbatch)
            dgp = denc["goal_pose"][0].float().cpu().numpy()
            d_disk = float(np.linalg.norm(dgp[:3] - disk[m][:3]))
            dd = dgp[:3] - cp[:3]
            b_disk = float(np.degrees(np.arctan2(dd[0], dd[2])))
        except Exception as exc:
            d_disk, b_disk = float("nan"), float("nan")
            if m == am:
                print(f"    (disk path unavailable: {exc})", flush=True)

        # 1. does goal_pose reproduce frame m's own recorded pose?
        dt = float(np.linalg.norm(gp[:3] - disk[m][:3]))

        # 2. the bearing the policy is effectively handed, vs geometric truth from the
        #    recorded poses. Bearing is what steers; a small metric error in the wrong
        #    direction still sends the robot the wrong way.
        d_model = gp[:3] - cp[:3]
        d_true = disk[m][:3] - disk[k][:3]
        b_model = float(np.degrees(np.arctan2(d_model[0], d_model[2])))
        b_true = float(np.degrees(np.arctan2(d_true[0], d_true[2])))
        err = abs((b_model - b_true + 180) % 360 - 180)
        derr = abs((b_disk - b_true + 180) % 360 - 180) if np.isfinite(b_disk) else float("nan")
        rows.append((dt, err, d_disk, derr))
        print(f"{m:5d} {dt:17.4f} {b_model:14.0f} {b_true:13.0f} {err:8.0f} "
              f"| {d_disk:9.4f} {derr:8.0f}", flush=True)

    on_err = np.mean([r[1] for r in rows])
    dk = [r[3] for r in rows if np.isfinite(r[3])]
    dk_err = np.mean(dk) if dk else float("nan")
    print("-" * 88, flush=True)
    print(f"\n  ONLINE bearing error  mean={on_err:.0f} deg", flush=True)
    print(f"  DISK   bearing error  mean={dk_err:.0f} deg", flush=True)
    print("\n=== VERDICT ===", flush=True)
    if not np.isfinite(dk_err):
        print("  disk path could not be evaluated -- online result stands alone.", flush=True)
    elif on_err > 30 and dk_err > 30:
        print("  BOTH paths are broken -> goal_append_warm itself is wrong, so TRAINING was", flush=True)
        print("  fit against broken goal geometry. Fix before spending any more GPU on it.", flush=True)
    elif on_err > 30 and dk_err <= 30:
        print("  ONLINE ONLY is broken -> the online cache I added is the culprit; training", flush=True)
        print("  is unaffected and the checkpoint is fine. Fix the online path.", flush=True)
    else:
        print("  Both paths look sound at these m -- pose is not the explanation.", flush=True)
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()

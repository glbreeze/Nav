#!/usr/bin/env python
"""Does deepening the warm-up remove the goal-pose error -- and is that the fix?

Measured so far: the goal-pose error tracks n_hist, the number of COMPRESSED history frames
injected before the warm window, and is unaffected by the KV sliding window (window=32 and
window=72 give the same error at the same n_hist). n_hist is

    n_hist = max(0, m - warm + 1 - num_scale)

so raising `warm` shrinks it, and once warm >= m - num_scale + 1 the warm-up covers the whole
history and NOTHING compressed is injected. Every row measured with n_hist == 0 had ~0 error.

This sweeps warm at fixed m. If the error collapses as n_hist -> 0, the mechanism is confirmed
(the 6-token-per-frame history compression cannot support goal pose insertion) and the fix is a
warm depth that reaches back far enough -- at a cost of `warm` frames of compute per step.
If the error does NOT collapse, the mechanism is something else and warm is a dead end.
"""
import argparse
import glob
import os
import sys
import time

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
    ap.add_argument("--ms", type=int, nargs="+", default=[105, 127, 171])
    ap.add_argument("--warms", type=int, nargs="+", default=[64, 96, 128, 176])
    args = ap.parse_args()
    dev = "cuda"

    cfg = memnav_exp_cfg.model_dump()
    cfg["il"]["root_dir"] = args.root_dir
    cfg["il"]["feature_root"] = args.feature_root
    for e, c in (("LINGBOT_REPO", "lingbot_repo"), ("LINGBOT_WEIGHTS", "lingbot_weights")):
        if os.environ.get(e):
            cfg["il"][c] = os.environ[e]
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
    disk = np.load(cam)["cam_pose_enc"]

    N = args.n_frames
    imgs = core.lingbot.load_images([os.path.join(rgb_dir, f"{i}.jpg") for i in range(N)])
    st = LingBotEpisodeState(core.lingbot, device=dev)
    for j in range(N):
        st.append_frame(imgs[j].to(dev))
    cache = st.build_cache()
    k = N - 1
    scale = core.num_scale

    print(f"\n[setup] k={k} num_scale={scale} (goal image = frame m itself, so the correct "
          f"answer is exactly that frame's own recorded pose)\n", flush=True)
    hdr = f"{'m':>5} {'warm':>6} {'n_hist':>7} {'pose err m':>11} {'bearing err':>12} {'ms':>7}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for m in args.ms:
        for warm in args.warms:
            start = max(scale, m - warm + 1)
            n_hist = max(0, start - scale)
            frames = torch.stack(st.frames[start:m + 1], 0)
            goal_img = imgs[m].to(dev)
            t0 = time.time()
            out = core.lingbot.goal_append_warm_frames(
                goal_img, cache, m, frames.to(dev), warm, return_agg=True)
            goal_agg = out[1] if isinstance(out, tuple) else out
            gp = core.lingbot.camera_pose(
                cache["cam_k"], cache["cam_v"], m + 1, goal_agg)[-1].float().cpu().numpy()
            torch.cuda.synchronize()
            ms_t = (time.time() - t0) * 1000

            cp = cache["cam_pose_enc"][k].float().cpu().numpy()
            dt = float(np.linalg.norm(gp[:3] - disk[m][:3]))
            dm, dtr = gp[:3] - cp[:3], disk[m][:3] - disk[k][:3]
            b_m = np.degrees(np.arctan2(dm[0], dm[2]))
            b_t = np.degrees(np.arctan2(dtr[0], dtr[2]))
            err = abs((b_m - b_t + 180) % 360 - 180)
            print(f"{m:5d} {warm:6d} {n_hist:7d} {dt:11.4f} {err:12.0f} {ms_t:7.0f}", flush=True)
        print("-" * len(hdr), flush=True)

    print("\n  Read the n_hist column: every row where it hits 0 should be ~0 error if the", flush=True)
    print("  compressed-history injection is the mechanism. The ms column is what that costs.", flush=True)
    print("\n=== SWEEP DONE ===", flush=True)


if __name__ == "__main__":
    main()

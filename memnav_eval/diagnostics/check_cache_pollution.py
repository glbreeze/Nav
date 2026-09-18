#!/usr/bin/env python
"""Does running the policy corrupt the live streaming KV cache?

LingBotEpisodeState.append_frame is autoregressive: the new frame attends to whatever sits in
self.agg.kv_cache, and _read_cache_anchor_newest() reads its stored KV straight back out of
that same cache. But every policy step then calls window_forward and goal_append_warm, and
BOTH begin with _inject() -> clean_kv_cache() -> refill from snapshot tensors. Neither saves
or restores what was there. stream_state has no save/restore anywhere (clean_kv_cache appears
only in __init__).

So the real eval loop is

    append_frame(k)        cache = live causal stream
    window_forward         cache = wiped, refilled, 32 frames replayed
    goal_append_warm       cache = wiped, refilled, 64 frames + goal at m+1
    append_frame(k+1)      <-- appends onto the GOAL-WARM leftovers, not the stream

while the earlier smoke test appended every frame in one uninterrupted loop and so never hit
this. That would leave retrieval intact (dino_cls is context-free, computed outside the cache)
while quietly corrupting cam_pose_enc and anchor_k -- which matches what eval shows:
match_hit=1.00 alongside a driving direction that is wrong even when n_hist is small.

Test: build the same stream twice over identical frames.
  A) clean   -- append only, nothing in between (what the smoke test did)
  B) polluted -- append, then a policy-shaped window_forward + goal_append_warm each step
Compare the resulting cam_pose_enc. Identical => the cache is fine and this is a dead end.
Divergent => every pose the policy reads during eval is wrong, independently of the n_hist
issue, and it gets worse the longer the episode runs.
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
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--policy_from", type=int, default=60,
                    help="start interleaving policy calls at this frame (needs k >= anchor_margin)")
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
    assert rgb_dir is not None
    disk = np.load(cam)["cam_pose_enc"]

    N = args.n_frames
    imgs = core.lingbot.load_images([os.path.join(rgb_dir, f"{i}.jpg") for i in range(N)])
    print(f"[traj] {rgb_dir}  N={N}  policy calls from frame {args.policy_from}\n", flush=True)

    # ---------------- A: clean ----------------
    st_a = LingBotEpisodeState(core.lingbot, device=dev)
    for j in range(N):
        st_a.append_frame(imgs[j].to(dev))
    pose_a = st_a.build_cache()["cam_pose_enc"].float().cpu().numpy()
    print(f"[A clean]    cam_pose_enc {pose_a.shape}", flush=True)

    # ---------------- B: policy calls interleaved ----------------
    st_b = LingBotEpisodeState(core.lingbot, device=dev)
    am = core.num_scale + core.window - 1
    for j in range(N):
        st_b.append_frame(imgs[j].to(dev))
        k = j
        if k < max(args.policy_from, am):
            continue
        # exactly what a policy step does to the cache, in the same order
        cache = st_b.build_cache()
        win = st_b.window_tensor(k, core.window).to(dev)
        core.lingbot.window_forward(cache, win, k, return_multilayer=True)
        m = max(am, min(k - 1, k // 2))
        core.lingbot.goal_append_warm_frames(
            imgs[m].to(dev), cache, m, st_b.warm_frames(m, core.goal_warm).to(dev),
            core.goal_warm, return_agg=True)
    pose_b = st_b.build_cache()["cam_pose_enc"].float().cpu().numpy()
    print(f"[B polluted] cam_pose_enc {pose_b.shape}\n", flush=True)

    # ---------------- compare ----------------
    n = min(len(pose_a), len(pose_b), len(disk))
    hdr = f"{'frame':>7} {'A vs disk':>11} {'B vs disk':>11} {'A vs B':>11}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    step = max(1, (n - args.policy_from) // 8)
    for f in list(range(args.policy_from, n, step)) + [n - 1]:
        da = np.abs(pose_a[f, :3] - disk[f, :3]).max()
        db = np.abs(pose_b[f, :3] - disk[f, :3]).max()
        ab = np.abs(pose_a[f, :3] - pose_b[f, :3]).max()
        print(f"{f:7d} {da:11.4f} {db:11.4f} {ab:11.4f}", flush=True)

    tail = slice(args.policy_from, n)
    ea = np.abs(pose_a[tail, :3] - disk[tail, :3]).max()
    eb = np.abs(pose_b[tail, :3] - disk[tail, :3]).max()
    print("-" * len(hdr), flush=True)
    print(f"\n  max |A - disk| = {ea:.4f} m   (clean stream vs precompute)", flush=True)
    print(f"  max |B - disk| = {eb:.4f} m   (policy-interleaved vs precompute)", flush=True)
    print("\n=== VERDICT ===", flush=True)
    if eb > max(10 * ea, 0.05):
        print("  CONFIRMED: policy calls corrupt the live stream. Every pose the policy reads", flush=True)
        print("  during eval is computed on a polluted cache -- separate from, and on top of,", flush=True)
        print("  the n_hist issue. Fix: save/restore the KV cache around window_forward and", flush=True)
        print("  goal_append_warm, or skip window_forward entirely and reuse the live cache", flush=True)
        print("  (it already holds the causal window -- recomputing 32 frames is redundant).", flush=True)
    else:
        print("  NOT corrupted: A and B agree, so the cache survives policy calls.", flush=True)
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()

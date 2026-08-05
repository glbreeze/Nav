#!/usr/bin/env python
"""Online-migration smoke: no habitat, no server. Feeds a REAL trajectory's frames through the
online stream and checks the plumbing the eval path depends on:

  (2) GOLD: online-built cam_pose_enc vs the precomputed disk cam_pose_enc. If these match, the
      stream reproduces precompute's continuous-stream pose exactly (the reason we capture the
      camera-head return per frame). A big diff = (2) is wrong, stop.
  (1)(3)(4): encode_memory + predict_action run on an online batch (online cache,
      goal_warm_frames, B from cur_steps) and return the right shapes.

Run inside the training apptainer/env. GPU needed (LingBot forward), tiny + fast.
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
    ap.add_argument("--n_frames", type=int, default=150, help=">122 so E(k) is non-empty")
    args = ap.parse_args()
    dev = "cuda"

    cfg = memnav_exp_cfg.model_dump()
    cfg["il"]["root_dir"] = args.root_dir
    cfg["il"]["feature_root"] = args.feature_root
    # explicitly wire the frozen LingBot repo/weights from env into the config -- the 14077632
    # run loaded 0 backbone weights (pretrained_path empty) because the config default didn't
    # pick them up, leaving the frozen encoder random and every readout garbage.
    if os.environ.get("LINGBOT_REPO"):
        cfg["il"]["lingbot_repo"] = os.environ["LINGBOT_REPO"]
    if os.environ.get("LINGBOT_WEIGHTS"):
        cfg["il"]["lingbot_weights"] = os.environ["LINGBOT_WEIGHTS"]
    assert cfg["il"].get("lingbot_weights"), "LINGBOT_WEIGHTS not set -- backbone would be random"
    policy = MemNavPolicy(MemNavModelConfig(model_cfg=cfg)).to(dev).eval()
    core = policy.core
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd if isinstance(sd, dict) else sd.state_dict()
    inc = policy.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(inc.missing_keys)} unexpected={len(inc.unexpected_keys)}", flush=True)

    # frames and caches live in SEPARATE trees: jpg under root_dir, cam_cache under
    # feature_root, same relative path. Resolve rgb_dir under root_dir -- NOT next to the
    # cam_cache (that dir holds only npz, no jpg).
    # jpg frames sit one dir deeper than the cam_cache: cam_cache is at .../videos/chunk-000/,
    # frames at .../videos/chunk-000/observation.images.rgb/{i}.jpg (loader's rgb_subdir).
    RGB_SUB = "observation.images.rgb"
    cam = rgb_dir = None
    for p in sorted(glob.glob(f"{args.feature_root}/mp3d_2leg/*/*/videos/chunk-000/lingbot_cam_cache.npz")):
        rel = os.path.relpath(p, args.feature_root)
        rd = os.path.join(args.root_dir, os.path.dirname(rel), RGB_SUB)
        if len(glob.glob(f"{rd}/*.jpg")) >= args.n_frames:
            cam, rgb_dir = p, rd
            break
    assert cam is not None, "no trajectory with enough frames found"
    disk_pose = np.load(cam)["cam_pose_enc"]
    print(f"[traj] {rgb_dir}  (disk cam_pose_enc {disk_pose.shape})", flush=True)

    N = args.n_frames
    imgs = core.lingbot.load_images([os.path.join(rgb_dir, f"{i}.jpg") for i in range(N)])
    st = LingBotEpisodeState(core.lingbot, device=dev)
    for j in range(N):
        st.append_frame(imgs[j].to(dev))
    assert st.ready()

    online_cache = st.build_cache()
    online_pose = online_cache["cam_pose_enc"].cpu().numpy()
    d = np.abs(online_pose - disk_pose[:N])
    print(f"[2 pose] online vs disk: max={d.max():.4f} mean={d.mean():.5f} "
          f"(translation cols0:3 max={d[:, :3].max():.4f})", flush=True)
    print(f"         online[0]={np.round(online_pose[0],3)}  disk[0]={np.round(disk_pose[0],3)}", flush=True)

    k = N - 1
    mem = st.mem_cls_tensor()
    goal_img = imgs[k].to(dev)
    goal_cls = core.lingbot.dino(goal_img.unsqueeze(0))["cls"][0]
    cand = torch.zeros(1, mem.shape[0], dtype=torch.bool, device=dev)
    am = core.num_scale + core.window - 1
    hi = k - int(os.environ.get("MEMNAV_EXCLUDE_RECENT", "83"))
    if hi >= am:
        cand[0, am:hi + 1] = True
    match_idx, _gl, _rl = core.retrieval(goal_cls.unsqueeze(0), mem.unsqueeze(0).to(dev), cand)
    m = int(match_idx[0].clamp(am, k - 1).item())
    print(f"[3 warm] k={k} m={m} E(k)=[{am}..{hi}] cand={int(cand.sum())}", flush=True)

    batch = {
        "batch_goal_cls": goal_cls.unsqueeze(0),
        "batch_mem_cls": mem.unsqueeze(0).to(dev),
        "batch_mem_mask": torch.ones(1, mem.shape[0], dtype=torch.bool),
        "batch_cand_mask": cand,
        "batch_goal_image": goal_img.unsqueeze(0),
        "batch_window_images": st.window_tensor(k, core.window).unsqueeze(0).to(dev),
        "batch_online_caches": [online_cache],
        "batch_goal_warm_frames": [st.warm_frames(m, core.goal_warm)],
        "cur_steps": [k],
    }

    enc = core.encode_memory(batch)
    print(f"[13 encode] cur_pose{tuple(enc['cur_pose'].shape)} goal_pose{tuple(enc['goal_pose'].shape)} "
          f"gate={float(enc['revisit_gate'][0]):.3f} match_idx={int(enc['match_idx'][0])}", flush=True)

    traj, vals, pos, neg = core.predict_action(batch, sample_num=2)
    print(f"[4 predict_action] traj{traj.shape} vals{vals.shape}  (expect traj (1,2,24,3))", flush=True)
    print("=== SMOKE OK ===", flush=True)


if __name__ == "__main__":
    main()

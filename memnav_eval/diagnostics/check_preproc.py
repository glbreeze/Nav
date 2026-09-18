#!/usr/bin/env python
"""Does the memory-frame preprocessing path diverge from the goal-image path enough to
break retrieval?

policy_agent feeds the two through DIFFERENT chains:

  goal   (set_goal / _goal_tensor):  raw -> _lingbot_frame -> 518x518
  memory (step_imagegoal:234):       raw -> process_image (cv2.resize to 308x168, /255)
                                         -> *255 -> _lingbot_frame -> 518x518

so every stored frame is squeezed to 308x168 and blown back up to 518, while the goal is
not. Training has no such split (both sides load full-res jpgs), which would make this a
pure train/inference inconsistency -- and it lines up with the observed eval failure
(gate=0.066, match_hit=0.0) against a clean profiler run (gate=0.612, match correct).

That is a code-reading argument. This measures it, and reports the one number that decides
whether it matters: does argmax_j cos(goal_i, mem_j) still land on j == i?
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream  # noqa: E402


def process_image(imgs, target_H=168, target_W=308):
    """Verbatim policy_agent.process_image."""
    out = []
    for img in imgs:
        resize = cv2.resize(img, (target_W, target_H))
        out.append(resize.astype(np.float32) / 255.0)
    return np.array(out)


def lingbot_frame(rgb_bgr, lb, tag):
    """Verbatim policy_agent._lingbot_frame."""
    rgb = cv2.cvtColor(
        (rgb_bgr * 255).astype(np.uint8) if rgb_bgr.max() <= 1.0 else rgb_bgr,
        cv2.COLOR_BGR2RGB)
    tmp = f"/tmp/preproc_{os.getpid()}_{tag}.jpg"
    cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    t = lb.load_images([tmp])[0]
    try:
        os.remove(tmp)
    except OSError:
        pass
    return t


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_dir", required=True)
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    dev = "cuda"

    lb = LingBotStream(
        lingbot_repo=os.environ["LINGBOT_REPO"], weights=os.environ["LINGBOT_WEIGHTS"],
        device=dev, num_scale=8, window=32, max_frame_num=2048).to(dev).eval()

    raw = []
    for i in range(args.n):
        p = os.path.join(args.rgb_dir, f"{i * 8}.jpg")   # spread out so frames differ
        im = cv2.imread(p)
        assert im is not None, f"cannot read {p}"
        raw.append(im)
    print(f"[imgs] {len(raw)} frames, native shape {raw[0].shape}", flush=True)
    print(f"[paths] goal: raw -> 518   |   memory: raw -> 308x168 -> 518\n", flush=True)

    A, B = [], []   # A = goal path, B = memory path
    for i, im in enumerate(raw):
        A.append(lb.dino(lingbot_frame(im, lb, f"a{i}").unsqueeze(0).to(dev))["cls"][0])
        proc = process_image([im])[0]
        B.append(lb.dino(lingbot_frame((proc * 255).astype(np.uint8), lb,
                                       f"b{i}").unsqueeze(0).to(dev))["cls"][0])
    A = torch.nn.functional.normalize(torch.stack(A), dim=-1)
    B = torch.nn.functional.normalize(torch.stack(B), dim=-1)

    same = (A * B).sum(-1)                       # same image, two paths
    cross_a = A @ A.T                            # different images, goal path
    off = ~torch.eye(len(A), dtype=torch.bool, device=dev)

    print("=== 1. same image through the two paths ===", flush=True)
    print(f"  cos(goal_path_i, mem_path_i): mean={same.mean():.4f} "
          f"min={same.min():.4f} max={same.max():.4f}", flush=True)
    print("\n=== 2. calibration: DIFFERENT images, same path ===", flush=True)
    print(f"  cos(goal_i, goal_j) i!=j     : mean={cross_a[off].mean():.4f} "
          f"max={cross_a[off].max():.4f}", flush=True)
    if same.mean() < cross_a[off].max():
        print("  !! the preprocessing gap is WIDER than the gap between different scenes", flush=True)

    print("\n=== 3. the decisive test: retrieval simulation ===", flush=True)
    print("  goal = A_i (goal path). memory = all B_j (memory path). does argmax land on i?", flush=True)
    S = A @ B.T
    hit = (S.argmax(-1) == torch.arange(len(A), device=dev))
    print(f"  argmax correct: {int(hit.sum())}/{len(A)}", flush=True)
    for i in range(min(8, len(A))):
        j = int(S[i].argmax())
        print(f"    goal {i:2d} -> picked mem {j:2d} "
              f"(cos={S[i, j]:.4f}, correct-one cos={S[i, i]:.4f}) "
              f"{'OK' if j == i else 'WRONG'}", flush=True)

    print("\n=== 4. control: if BOTH sides used the goal path ===", flush=True)
    hit2 = (cross_a.argmax(-1) == torch.arange(len(A), device=dev))
    print(f"  argmax correct: {int(hit2.sum())}/{len(A)}", flush=True)

    print("\n=== VERDICT ===", flush=True)
    if int(hit.sum()) < len(A) and int(hit2.sum()) == len(A):
        print("  CONFIRMED: retrieval breaks ONLY when the two paths differ.", flush=True)
        print("  Fix = feed the raw image to _lingbot_frame in step_imagegoal/stream_frame.", flush=True)
    elif int(hit.sum()) == len(A):
        print("  NOT the cause: retrieval survives the preprocessing gap. Look elsewhere.", flush=True)
    else:
        print("  INCONCLUSIVE: retrieval fails even with matched paths -- something else too.", flush=True)


if __name__ == "__main__":
    main()

# MemNav — Implicit-Memory Image-Goal Navigation

MemNav is an image-goal navigation policy with **persistent implicit memory**. It replaces
NavDP's shallow 8-frame RGBD window with a frozen **LingBot-Map** streaming backbone
(Geometric Context Transformer, `/home/asus/Research/lingbot-map/`) as the history encoder, and
conditions on a **single RGB goal image** matched against that history by a trainable
**cross-view retrieval head**. The trajectory planner reuses NavDP/LoGoPlanner's DDPM diffusion
decoder + critic.

The design target is **seen ≫ unseen** on the multi-stop image-goal benchmark: when the goal
location was already observed earlier in the episode, memory should let the policy exploit it;
NavDP (memoryless) is the floor.

## Architecture (inference path)

```
                goal RGB ───► LingBotStream.encode_goal ──► dino_desc[1024], dino_patch[P,1024]   (context-free)
                                                                   │
 current RGB ──► LingBotStream.step ──► per-frame:                 │   (frozen DINOv2 + GCT aggregator)
                   anchor[6,2048]        (pose/geometry, all frames)│
                   dino_desc[1024]       (match key, all frames)    │
                   dino_patch[P,1024]    (dense match, window frames)│
                          │                                         ▼
                          └──────────────► CrossViewRetrievalHead ──► goal-grounded context tokens
                                            (coarse frame retrieval                  │
                                             → dense patch cross-view)               ▼
                                                            MemNav diffusion decoder + critic
                                                                       │
                                                                       ▼
                                                               local trajectory (24×3)
```

Key facts settled during design (see project memory / `multistop_benchmark_design.md`):
- **Frozen LingBot, RGB only** (no depth). Live KV-cache streaming at inference matches the
  offline precompute used for training (`InternNav/scripts/dataset_converters/precompute_lingbot_features.py`).
- **Three LingBot tiers** are native to the backbone: scale frames (full patches), sliding
  window of `kv_cache_sliding_window=64` recent frames (full patches), older frames compressed to
  6 special/anchor tokens. The per-frame **output tokens are write-once**, so the streamed
  features here are identical to the cached ones used in training.
- **Matching space** = context-free DINOv2 patches (`dino_desc` pooled / `dino_patch` full).
  Goal and history use the *same* frozen encoder → symmetric. Dense cross-view needs full-res
  patches (no pooling) for correspondence; the head's *output* is compressed before the decoder.
- **Geometry/value space** = `anchor` (6 special tokens: camera + 4 register + scale), pose-
  grounded in LingBot's drift-corrected frame.

## Files

| file | role |
|---|---|
| `policy_backbone.py` | reused diffusion blocks (`SinusoidalPosEmb`, `LearnablePositionalEncoding`, `TokenCompressor`) + `LingBotStream` (frozen live feature extractor) |
| `policy_network.py`  | `CrossViewRetrievalHead` + `MemNav_Policy` (retrieval head + DDPM decoder + critic) |
| `policy_agent.py`    | `MemNav_Agent` — owns `LingBotStream` + `MemNav_Policy`, keeps streaming state across steps, encodes the goal once per episode |
| `memnav_server.py`   | Flask server: `/navigator_reset`, `/navigator_reset_env`, `/imagegoal_step` |
| `requirements.txt`   | server env deps (torch 2.8 + lingbot-map + diffusers + flask) |

The **training** counterpart lives in the sibling repo: `InternNav/internnav/model/basemodel/memnav/`,
`InternNav/internnav/trainer/memnav_trainer.py`, `InternNav/internnav/dataset/memnav_dataset_lerobot.py`.
The model definition (`MemNav_Policy`, `CrossViewRetrievalHead`) is mirrored between the two repos
(same convention as `navdp`/`logoplanner`); keep them in sync.

## Setup — vendored dependencies (not in this repo)

Two third-party repos are **gitignored** (their weights exceed GitHub's 100 MB limit), so a fresh
clone of `Nav` has empty `lingbot-map/` and `Long-CLIP/` dirs. Re-create them:

```bash
# 1. LingBot-Map — frozen GCT streaming backbone (pinned commit)
git clone https://github.com/Robbyant/lingbot-map.git \
    NavDP/baselines/memnav/lingbot-map
git -C NavDP/baselines/memnav/lingbot-map checkout 3c63739

# 2. LingBot weights (~4.4 GB) — download into the repo's weights/ dir.
#    Path must match LingBotStream's default `weights=` arg:
#      NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt
#    (obtain from the LingBot-Map release / project owner)

# 3. Long-CLIP — only needed by the navdp model path (pinned commit)
git clone https://github.com/beichenzbc/Long-CLIP.git \
    InternNav/internnav/model/basemodel/Long-CLIP
git -C InternNav/internnav/model/basemodel/Long-CLIP checkout 3966af9
```

Validate the LingBot streaming wrapper after setup (expects cosine ≈ 1.0 vs the official forward):

```bash
conda activate memnav
python InternNav/scripts/dataset_converters/validate_lingbot_stream.py
```

## Run

```bash
conda activate memnav        # torch 2.8 + lingbot-map + diffusers + flask
cd NavDP/baselines/memnav/
python memnav_server.py --port 8888 \
    --checkpoint ./checkpoints/memnav.ckpt \
    --lingbot_repo /home/asus/Research/lingbot-map \
    --lingbot_weights /home/asus/Research/lingbot-map/weights/lingbot-map-long.pt
```

then, from `NavDP/` (IsaacSim env):

```bash
python eval_imagegoal_wheeled.py --port 8888 --scene_dir /abs/path/to/cluttered_hard --scene_index 0 --scene_scale 1.0
```

> Image-goal renders black on `internscenes_home` — use `cluttered_hard`. Run one IsaacSim eval
> at a time. (project memory: multistop-imagegoal-scene)

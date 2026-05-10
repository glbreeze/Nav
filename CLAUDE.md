# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This working directory contains two sibling repositories:
- `NavDP/` — inference + evaluation (this is the benchmark suite)
- `InternNav/` — training code for NavDP models

NavDP (Navigation Diffusion Policy) is an end-to-end mapless navigation benchmark and model suite built on NVIDIA IsaacSim/IsaacLab. It evaluates visual navigation methods via a decoupled client-server architecture: navigation models run as Flask HTTP servers, while IsaacSim runs the simulation environment and sends RGB-D observations to the server for trajectory planning.

**Training lives in the sibling `InternNav/` directory.** The `NavDP/` repo is inference + evaluation only. `InternNav/` is a broader navigation toolbox covering several model families — `cma`, `seq2seq`, `rdp`, `navdp`, `logoplanner`, `internvla_n1` — each with parallel `internnav/trainer/<m>_trainer.py`, `internnav/dataset/<m>_dataset_lerobot.py`, `internnav/model/basemodel/<m>/`, and `scripts/train/configs/<m>.py`. The dispatch happens in `scripts/train/train.py` via `--model-name`. When the user asks about training, dataset loaders, loss functions, or the InternData-N1 dataset format, read from `InternNav/` rather than `NavDP/`.

`GL.md` (root and `NavDP/GL.md`) is the user's running scratchpad with dataset conversion notes and paper-derived loss/training plans — useful context but not authoritative; trust the code.

## Environment Setup

Two separate conda environments are required:

**Benchmark environment** (IsaacSim evaluation):
```bash
conda create -n isaaclab python=3.10
pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com
# Then install IsaacLab v1.2.0 per README
cd NavDP/
pip install -r requirements.txt
```

**Model server environment** (per-baseline, e.g. NavDP):
```bash
conda create -n navdp python=3.10
cd NavDP/baselines/navdp/
pip install -r requirements.txt
```

Some baselines (DD-PPO, ViPlanner, GNM/ViNT/NoMad) require their own environments with additional dependencies — see README for details.

## Running the System

The workflow is always: start a model server, then run evaluation or teleoperation against it.

**Start a baseline server** (each baseline has a `*_server.py`):
```bash
cd NavDP/baselines/navdp/
python navdp_server.py --port 8888 --checkpoint ./checkpoints/navdp_checkpoint.ckpt
```

**Run evaluation** (requires IsaacSim environment, run from `NavDP/`). One script per task type — `nogoal`, `pointgoal`, `imagegoal`, `startgoal` (start→goal without odometry):
```bash
cd NavDP/
python eval_pointgoal_wheeled.py --port 8888 --scene_dir /absolute/path/to/scene --scene_index 0 --scene_scale 1.0
python eval_nogoal_wheeled.py    --port 8888 --scene_dir /absolute/path/to/scene --scene_index 0 --scene_scale 1.0
python eval_imagegoal_wheeled.py --port 8888 --scene_dir /absolute/path/to/scene --scene_index 0 --scene_scale 1.0
python eval_startgoal_wheeled.py --port 8888 --scene_dir /absolute/path/to/scene --scene_index 0 --scene_scale 1.0
```
Scene scale: `0.01` for internscenes, `1.0` for cluttered scenes. Scene dir must be an absolute path. `run.sh` at the NavDP root has a working LoGoPlanner end-to-end smoke test (server on port 19999 → `eval_startgoal_wheeled.py`).

**Teleoperation** (keyboard control with trajectory visualization):
```bash
cd NavDP/
python teleop_pointgoal_wheeled.py   # WASD controls
```

**InternNav training** (run from `InternNav/`). `scripts/train/start_train.sh` is the canonical wrapper — it sets `CUDA_VISIBLE_DEVICES` per model and uses `torchrun` for `navdp` (8-GPU) but plain `python` otherwise:
```bash
bash scripts/train/start_train.sh --name my_run --model rdp        # cma | cma_plus | seq2seq | seq2seq_plus | rdp | navdp
bash scripts/train/runs.sh                                         # single-GPU smoke test for logoplanner on a small shard
```
The `rdp`/`navdp`/`logoplanner` paths import from `src/diffusion-policy/` — that's a git submodule (`git submodule update --init`) and `runs.sh` exports `PYTHONPATH="$PWD/src/diffusion-policy:..."`. If you launch via `train.py` directly, replicate that export.

**InternNav evaluation**: `scripts/eval/eval_habitat.py` for VLN-CE / Habitat tasks; `scripts/eval/eval.py` + `scripts/eval/start_server.py` (with configs in `scripts/eval/configs/h1_*_cfg.py`) for IsaacSim/InternUtopia tasks. InternNav-trained checkpoints can also be served to `NavDP/`'s eval scripts via the matching baseline server in `NavDP/baselines/`.

## Architecture

### Client-Server Decoupling

Navigation methods are decoupled from the simulation via HTTP. The simulation (IsaacSim) acts as a client sending RGB-D observations; the navigation model acts as a Flask server returning planned trajectories. This design lets models run in separate environments/GPUs.

- **Client side**: `NavDP/utils_tasks/client_utils.py` — sends observations to the server via HTTP POST (images as encoded bytes, depth as uint16 PNG)
- **Server side**: Each baseline's `*_server.py` — Flask app exposing `/navigator_reset`, `/{task}_step` endpoints (e.g., `/pointgoal_step`, `/nogoal_step`, `/imagegoal_step`)

### Evaluation Scripts (`NavDP/` root)

`NavDP/eval_{task}_wheeled.py` and `NavDP/teleop_{task}_wheeled.py` — one per task type (nogoal, pointgoal, imagegoal, startgoal). These launch IsaacSim, set up the scene, spawn the robot, and run an async planning thread that calls the navigation server while an MPC controller tracks the planned trajectory.

### Baselines (`NavDP/baselines/`)

Each subdirectory is a self-contained navigation method: `navdp`, `logoplanner`, `iplanner`, `viplanner`, `ddppo`, `gnm`, `vint`, `nomad`. Each contains:
- `*_server.py` — Flask server with standard HTTP API
- `*_agent.py` — wraps the model for inference (manages observation history, preprocessing)
- `policy_network.py` / `*_network.py` — the neural network model

**NavDP model** (`NavDP/baselines/navdp/`): Uses DepthAnythingV2 as the RGB-D backbone, a transformer decoder for temporal fusion, and DDPM diffusion (10 timesteps) to generate trajectory predictions. Key params: `memory_size=8` (observation history), `predict_size=24` (waypoints), `token_dim=384`.

**LoGoPlanner** (`NavDP/baselines/logoplanner/`): The newer localization-grounded extension of NavDP, with its own `README.md`, real-world deployment code (`lekiwi_logoplanner_host.py`, `deployment/`) and a Pi3-based geometry head (`geometry_model.py`, vendored `Pi3/` directory). Demonstrated end-to-end in `NavDP/run.sh`.

### Simulation Framework (`NavDP/configs/`, `NavDP/wheeled_robots/`)

- `NavDP/configs/robots/` — robot configuration (Dingo wheeled robot)
- `NavDP/configs/scenes/` — scene loading and USD asset configuration
- `NavDP/configs/tasks/wheeled_task.py` — IsaacLab `ManagerBasedRLEnvCfg` defining observations (camera RGB/depth), rewards, terminations
- `NavDP/wheeled_robots/controllers/` — differential drive controller
- `NavDP/utils_tasks/tracking_utils.py` — MPC-based trajectory tracking controller
- `NavDP/utils_tasks/visualization_utils.py` — real-time trajectory visualization

### Scene Assets

Downloaded separately from HuggingFace (`InternRobotics/Scene-N1`). Four scene categories: `cluttered_easy`, `cluttered_hard`, `internscenes_home`, `internscenes_commercial`. Each scene directory contains USD files and episode definitions (`pointgoal_start_goal_pairs.npy`, `imagegoal_start_goal_pairs.npy`).

## Key Conventions

- All baseline servers follow the same HTTP API contract — new baselines should implement the same endpoints
- The planning loop runs asynchronously in a separate thread from the simulation step loop
- Depth images are transmitted as uint16 PNG (multiplied by 10000) for precision
- Trajectories are represented as arrays of 3D waypoints (x, y, heading) in the robot's local frame

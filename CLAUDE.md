# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This working directory contains two sibling repositories:
- `NavDP/` — inference + evaluation (this is the benchmark suite)
- `InternNav/` — training code for NavDP models

NavDP (Navigation Diffusion Policy) is an end-to-end mapless navigation benchmark and model suite built on NVIDIA IsaacSim/IsaacLab. It evaluates visual navigation methods via a decoupled client-server architecture: navigation models run as Flask HTTP servers, while IsaacSim runs the simulation environment and sends RGB-D observations to the server for trajectory planning.

**Training lives in the sibling `InternNav/` directory.** The `NavDP/` repo is inference + evaluation only. NavDP model training code is in `InternNav/` — entry `scripts/train/train.py --model-name navdp`, trainer `internnav/trainer/navdp_trainer.py`, dataset `internnav/dataset/navdp_dataset_lerobot.py`, config `scripts/train/configs/navdp.py`. When the user asks about training, dataset loaders, loss functions, or the InternData-N1 dataset format, read from `InternNav/` rather than `NavDP/`.

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

**Run evaluation** (requires IsaacSim environment, run from `NavDP/`):
```bash
cd NavDP/
python eval_pointgoal_wheeled.py --port 8888 --scene_dir /absolute/path/to/scene --scene_index 0 --scene_scale 1.0
python eval_nogoal_wheeled.py --port 8888 --scene_dir /absolute/path/to/scene --scene_index 0 --scene_scale 1.0
python eval_imagegoal_wheeled.py --port 8888 --scene_dir /absolute/path/to/scene --scene_index 0 --scene_scale 1.0
```
Scene scale: `0.01` for internscenes, `1.0` for cluttered scenes. Scene dir must be an absolute path.

**Teleoperation** (keyboard control with trajectory visualization):
```bash
cd NavDP/
python teleop_pointgoal_wheeled.py   # WASD controls
```

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

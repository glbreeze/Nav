#!/bin/bash
#SBATCH --job-name=memnav_eval_w32
#SBATCH --account=torch_pr_926_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ay2710/logs/memnav_eval_w32_%j.log
#
# Closed-loop image-goal nav eval in HABITAT MP3D (same domain as training).
# Single GPU: memnav server (lbm env, online cache) + habitat eval (habitat env).
#
# Everything now runs against OUR checkout of the professor's code, not the peer tree:
#   - model code : Nav_memnav/InternNav @ 5dcc22a (decoupled loss / E(k) / teacher forcing),
#                  plus our two additions the professor's code lacks for closed loop --
#                  online-cache inputs and predict_action (she only ships forward()).
#   - lingbot-map: the professor's CLEAN checkout. Her rope fix lives in InternNav
#                  (rebuilds the table to max_frame_num=2048), so a clean repo is correct.
#                  The peer repo's `% len` rope hack WRAPS the positional encoding past
#                  frame 1024 instead of extending it -- do not put it back on PYTHONPATH.
#
# MEMNAV_WINDOW must match how the checkpoint's caches were precomputed (w32 run: 32).
# One export drives both the server (memory_size) and the config (LingBotStream.window).
#
# MEMNAV_SWAP_XY defaults to 1, not 0: the habitat client's fwd_xz/left_xz read the yaw
# axes transposed, and without the swap the robot spins in place and every episode scores 0.
# This was diagnosed the hard way -- 0 is not a safe default here.
set -uo pipefail
JOB=${SLURM_JOB_ID}
LOGDIR=/scratch/ay2710/logs/memnav_eval_w32_${JOB}
mkdir -p "${LOGDIR}"
MFF=/scratch/ay2710/hrdt/miniforge3/etc/profile.d/conda.sh
source "${MFF}"

PORT=${PORT:-28890}
CKPT=${MEMNAV_CKPT:?set MEMNAV_CKPT to the checkpoint to evaluate}
NEPS=${NUM_EPISODES:-50}
MAX_STEPS=${MAX_STEPS:-200}
STOP_THRESH=${STOP_THRESH:--1000}   # very low -> server never forces the stop placeholder
SCENE=${SCENE:-/scratch/ay2710/mp3d_scenes/17DRP5sb8fy/17DRP5sb8fy.glb}

OUR_INTERNNAV=/scratch/ay2710/Nav_memnav/InternNav
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
LINGBOT_W=${LINGBOT_REPO}/weights/lingbot-map-long.pt
export MEMNAV_WINDOW=${MEMNAV_WINDOW:-32}
export MEMNAV_EXCLUDE_RECENT=${MEMNAV_EXCLUDE_RECENT:-83}
# classifier-free guidance scale. 1.0 = plain conditional (historical behavior, no extra
# forward). >1 leans harder on the goal; the ng branch it needs is already trained.
export MEMNAV_CFG_W=${MEMNAV_CFG_W:-1.0}
FWD_SIGN=${MEMNAV_FWD_SIGN:-1}; LAT_SIGN=${MEMNAV_LAT_SIGN:-1}; SWAP_XY=${MEMNAV_SWAP_XY:-1}

echo "=== launch server (lbm) ckpt=${CKPT} window=${MEMNAV_WINDOW} ==="
CUDA_VISIBLE_DEVICES=0 bash -c "
  source ${MFF}; conda activate lbm
  export MEMNAV_SAMPLE_NUM=${SAMPLE_NUM:-8}
  export MEMNAV_WINDOW=${MEMNAV_WINDOW} MEMNAV_EXCLUDE_RECENT=${MEMNAV_EXCLUDE_RECENT}
  export MEMNAV_CFG_W=${MEMNAV_CFG_W}
  export PYTHONPATH=${OUR_INTERNNAV}:${OUR_INTERNNAV}/src/diffusion-policy:${LINGBOT_REPO}
  cd /scratch/ay2710/Nav_memnav/NavDP/baselines/memnav
  python memnav_server.py \
    --checkpoint ${CKPT} \
    --lingbot_repo ${LINGBOT_REPO} \
    --lingbot_weights ${LINGBOT_W} \
    --vid_dir ${LOGDIR} --port ${PORT}
" > "${LOGDIR}/server.log" 2>&1 &
SERVER_PID=$!

echo "=== wait server ready (max 6min) ==="
for i in $(seq 1 180); do
  grep -q "Running on http" "${LOGDIR}/server.log" 2>/dev/null && { echo "server ready (${i}x2s)"; break; }
  kill -0 ${SERVER_PID} 2>/dev/null || { echo "SERVER DIED"; tail -40 "${LOGDIR}/server.log"; exit 1; }
  sleep 2
done

echo "=== run HABITAT image-goal eval (habitat env) SWAP_XY=${SWAP_XY} ==="
CUDA_VISIBLE_DEVICES=0 bash -c "
  source ${MFF}; conda activate /scratch/ay2710/hrdt/miniforge3/envs/habitat
  export MEMNAVDATA_DIR=/scratch/ay2710/Nav_memnav/MemNavData
  export MEMNAV_FWD_SIGN=${FWD_SIGN} MEMNAV_LAT_SIGN=${LAT_SIGN} MEMNAV_SWAP_XY=${SWAP_XY}
  export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
  cd /scratch/ay2710/Nav_memnav/NavDP
  python -u eval_imagegoal_habitat.py \
    --scene ${SCENE} --port ${PORT} --num_episodes ${NEPS} --max_steps ${MAX_STEPS} \
    --stop_threshold ${STOP_THRESH} --d_min ${D_MIN:-1.5} --d_max ${D_MAX:-6.0} \
    --success_dist ${SUCCESS_DIST:-1.0} --cam_h ${CAM_H:-0.5} \
    --out ${LOGDIR}/hab_out --save_video
" 2>&1 | tee "${LOGDIR}/eval.log"
EVAL_EXIT=${PIPESTATUS[0]}

echo "=== cleanup (eval exit=${EVAL_EXIT}) ==="
kill ${SERVER_PID} 2>/dev/null; sleep 3; kill -9 ${SERVER_PID} 2>/dev/null
echo "=== result ==="
cat ${LOGDIR}/hab_out/metrics.json 2>/dev/null | tail -20
echo "logs: ${LOGDIR}/server.log ; ${LOGDIR}/eval.log ; videos: ${LOGDIR}/hab_out/"

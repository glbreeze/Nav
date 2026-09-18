#!/bin/bash
#SBATCH --job-name=memnav_revisit
#SBATCH --account=torch_pr_926_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ay2710/logs/memnav_revisit_%j.log
#
# Revisit vs novel image-goal eval: the experiment the plain eval cannot run.
#
# eval_imagegoal_habitat.py samples random start/goal, so the goal is always somewhere the
# robot has never been -> gate says novel -> the revisit tokens are masked out -> the memory
# is never used. Its SR measures the novel branch alone, which is the part MemNav shares
# with any image-goal policy. This driver walks the robot PAST the goal first (leg 1),
# walks it away (leg 2, so the goal clears the exclude_recent window and lands inside E(k)),
# and only then asks the policy to go back -- and pairs every such episode with a matched
# novel goal that is provably clear of the traversed path.
#
# Reports SR_revisit vs SR_novel, plus two measurements SR cannot give you:
#   gate_sep   = P(revisit) on revisit runs - on novel runs   (is the gate deciding at all?)
#   match_hit  = did retrieval return a frame actually near the goal (ground truth)
# SR sits downstream of MPC/collision/trajectory-choice; these two are the direct readout.
set -uo pipefail
JOB=${SLURM_JOB_ID}
LOGDIR=/scratch/ay2710/logs/memnav_revisit_${JOB}
mkdir -p "${LOGDIR}"
MFF=/scratch/ay2710/hrdt/miniforge3/etc/profile.d/conda.sh
source "${MFF}"

PORT=${PORT:-28899}
CKPT=${MEMNAV_CKPT:?set MEMNAV_CKPT to the checkpoint to evaluate}
NEPS=${NUM_EPISODES:-20}
SCENE=${SCENE:-/scratch/ay2710/mp3d_scenes/17DRP5sb8fy/17DRP5sb8fy.glb}

OUR_INTERNNAV=/scratch/ay2710/Nav_memnav/InternNav
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
LINGBOT_W=${LINGBOT_REPO}/weights/lingbot-map-long.pt
export MEMNAV_WINDOW=${MEMNAV_WINDOW:-32}
export MEMNAV_EXCLUDE_RECENT=${MEMNAV_EXCLUDE_RECENT:-83}
export MEMNAV_CFG_W=${MEMNAV_CFG_W:-1.0}
SWAP_XY=${MEMNAV_SWAP_XY:-1}

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
  grep -q "Running on http" "${LOGDIR}/server.log" 2>/dev/null && { echo "server ready"; break; }
  kill -0 ${SERVER_PID} 2>/dev/null || { echo "SERVER DIED"; tail -40 "${LOGDIR}/server.log"; exit 1; }
  sleep 2
done

echo "=== revisit vs novel eval (habitat env) SWAP_XY=${SWAP_XY} ==="
CUDA_VISIBLE_DEVICES=0 bash -c "
  source ${MFF}; conda activate /scratch/ay2710/hrdt/miniforge3/envs/habitat
  export MEMNAVDATA_DIR=/scratch/ay2710/Nav_memnav/MemNavData
  export MEMNAV_SWAP_XY=${SWAP_XY}
  export MEMNAV_EXCLUDE_RECENT=${MEMNAV_EXCLUDE_RECENT}
  # leg-3 return distance band. The default upper bound is 6 m, but the image-goal eval
  # only ever succeeds inside 1.5-3.0 m (successes at 1.5/2.0/2.0/2.2/2.3 m, failures from
  # 1.8 m up), so sampling out to 6 m makes BOTH conditions fail and SR_gain uninformative.
  export MEMNAV_DBACK_LO=${MEMNAV_DBACK_LO:-1.5}
  export MEMNAV_DBACK_HI=${MEMNAV_DBACK_HI:-3.0}
  export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
  cd /scratch/ay2710/Nav_memnav/NavDP
  python -u eval_revisit_habitat.py \
    --scene ${SCENE} --port ${PORT} --num_episodes ${NEPS} \
    --max_steps ${MAX_STEPS:-300} --success_dist ${SUCCESS_DIST:-1.0} \
    --out ${LOGDIR}/revisit_out --save_video --vid_stride ${VID_STRIDE:-2}
" 2>&1 | tee "${LOGDIR}/eval.log"
EVAL_EXIT=${PIPESTATUS[0]}

echo "=== cleanup (eval exit=${EVAL_EXIT}) ==="
kill ${SERVER_PID} 2>/dev/null; sleep 3; kill -9 ${SERVER_PID} 2>/dev/null
echo "=== result ==="
cat ${LOGDIR}/revisit_out/metrics.json 2>/dev/null
echo "logs: ${LOGDIR}/server.log ; ${LOGDIR}/eval.log"

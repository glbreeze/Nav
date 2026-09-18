#!/bin/bash
#SBATCH --job-name=memnav_pollute
#SBATCH --account=torch_pr_926_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/ay2710/logs/memnav_pollute_%j.log
# Where does the 11.4 s/step actually go, and is match_idx stable enough to cache goal_agg?
# Both are currently guesses in the md. Short wall time so backfill can slot it in.
set -uo pipefail
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
INTERNNAV=/scratch/ay2710/Nav_memnav/InternNav
CKPT=${MEMNAV_CKPT:?set MEMNAV_CKPT}
export MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 MEMNAV_EXCLUDE_RECENT=83
export LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
export LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt

echo "host: $(hostname); ckpt=${CKPT}"
singularity exec --nv -B /scratch/lg154 -B /scratch/ay2710 "${SIF}" bash -c '
    set -uo pipefail
    source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
    conda activate /scratch/lg154/conda-envs/memnav
    cd "'"${INTERNNAV}"'"
    export PYTHONPATH="'"${INTERNNAV}"'/src/diffusion-policy:${PYTHONPATH:-}"
    python -u /scratch/ay2710/check_cache_pollution.py --ckpt "'"${CKPT}"'"
'
echo "=== done ==="

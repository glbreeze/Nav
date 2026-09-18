#!/bin/bash
#SBATCH --job-name=memnav_preproc
#SBATCH --account=torch_pr_926_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/ay2710/logs/memnav_preproc_%j.log
# Does the memory-frame preprocessing chain (raw -> 308x168 -> 518) diverge from the
# goal-image chain (raw -> 518) enough to break retrieval? Decides whether gate=0.066 /
# match_hit=0.0 in the revisit eval is a plumbing bug or a model failure. Tiny + short.
set -uo pipefail
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
INTERNNAV=/scratch/ay2710/Nav_memnav/InternNav
RGB=${RGB_DIR:-/scratch/ay2710/mp3d_data/vln_n1/traj_data/mp3d_2leg/17DRP5sb8fy/episode_0000/videos/chunk-000/observation.images.rgb}
export LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
export LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt

echo "host: $(hostname); rgb=${RGB}"
singularity exec --nv -B /scratch/lg154 -B /scratch/ay2710 "${SIF}" bash -c '
    set -uo pipefail
    source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
    conda activate /scratch/lg154/conda-envs/memnav
    cd "'"${INTERNNAV}"'"
    export PYTHONPATH="'"${INTERNNAV}"'/src/diffusion-policy:${PYTHONPATH:-}"
    python -u /scratch/ay2710/check_preproc.py --rgb_dir "'"${RGB}"'"
'
echo "=== done ==="

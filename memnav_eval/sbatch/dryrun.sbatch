#!/bin/bash
#SBATCH --job-name=memnav_dryrun
#SBATCH --account=torch_pr_926_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/ay2710/logs/memnav_dryrun_%j.log
# Sampling only: can the generator's own rules be satisfied in this scene, and what
# distribution do they give? No server, no policy, no GPU inference -- habitat rendering
# only. Catches a bad episode spec in minutes instead of after hours of navigation.
set -uo pipefail
MFF=/scratch/ay2710/hrdt/miniforge3/etc/profile.d/conda.sh
SCENE=${SCENE:-/scratch/ay2710/mp3d_scenes/17DRP5sb8fy/17DRP5sb8fy.glb}
echo "host: $(hostname); scene: ${SCENE}"
bash -c "
  source ${MFF}; conda activate /scratch/ay2710/hrdt/miniforge3/envs/habitat
  export MEMNAVDATA_DIR=/scratch/ay2710/Nav_memnav/MemNavData
  export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
  cd /scratch/ay2710/Nav_memnav/NavDP
  python -u eval_revisit_habitat.py --scene ${SCENE} --num_episodes ${NEPS:-8} --dry_run
"
echo "=== done ==="

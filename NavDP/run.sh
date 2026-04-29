
cd /home/asus/Research/NavDP/baselines/logoplanner 
source /home/asus/miniconda3/etc/profile.d/conda.sh 
conda activate navdp 

# Start the server
cd baselines/logoplanner
python logoplanner_server.py --port 19999 --checkpoint ./ckpt/logoplanner_policy.ckpt 2>&1

# data for smoke test
# https://huggingface.co/datasets/InternRobotics/Scene-N1


# Evaluate on scenes_home
conda activate isaaclab 
python eval_startgoal_wheeled.py --port 19999 --scene_dir /home/asus/Research/NavDP/assets/scenes/internscenes_home --scene_index 0 --scene_scale 0.01

# Evaluate on cluttered_hard
python eval_startgoal_wheeled.py --port 19999 --scene_dir /home/asus/Research/NavDP/assets/scenes/cluttered_hard --scene_index 0 --scene_scale 1.0
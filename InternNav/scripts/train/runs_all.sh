conda activate navdp

python scripts/train/train.py \
    --name test \
    --model-name logoplanner \
    --batch-size 2 \
    --num-workers 0 \
    --epochs 1 \
    --root-dir /home/asus/Research/datasets/InternData-N1/vln_n1/traj_data_navdp \
    --dataset-navdp /tmp/logoplanner_dataset_lerobot.json

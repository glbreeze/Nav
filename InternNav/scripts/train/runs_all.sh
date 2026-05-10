conda activate navdp

# activate conda env on Torch
cd /scratch/lg154/Research/Nav/InternNav                                                                                                                                                        
conda activate /scratch/lg154/conda-envs/navdp 


export PYTHONPATH="$PWD/src/diffusion-policy:${PYTHONPATH:-}"                                                                                                                                   
export CUDA_VISIBLE_DEVICES=0
                                                                                                                                                                                                  
python scripts/train/train.py \
    --name test \
    --model-name logoplanner \
    --batch-size 2 \
    --num-workers 0 \
    --epochs 1 \
    --root-dir /scratch/lg154/Research/datasets/InternData-N1/vln_n1/_raw \
    --dataset-navdp /tmp/logoplanner_dataset_lerobot.json         
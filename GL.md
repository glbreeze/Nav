                                                                           # Data preprocessing     
  added New files:                                                                                                          
  - /home/asus/Research/datasets/InternData-N1/convert_navdp.py — the converter
  - .../vln_n1/_raw/replica_zed/apartment_1/ — raw tarball extraction (~1 GB)                                      
  - .../vln_n1/traj_data_navdp/replica_zed/apartment_1/ep_000…ep_017/ — 18 converted traj dirs (mostly symlinks + small generated path.ply per episode)
                                                                                                                                                                                                   
  Loader patch (InternNav/internnav/dataset/navdp_dataset_lerobot.py:177):                                                                                                                         
  - before: camera_trajectory = np.array([np.stack(frame) for frame in df['action']], dtype=np.float64)                                                                                            
  - after:  ... ).reshape(-1, 4, 4)                                                                                                                                                                
# Run training                                                                                                                                                                                                                                                 
  Edit InternNav/scripts/train/configs/navdp.py:                                                                                                                                                   
  - root_dir='/home/asus/Research/datasets/InternData-N1/vln_n1/traj_data_navdp'                                                                                                                   
  - dataset_navdp='/tmp/navdp_cache/apartment_1.json'                                                                                                                                              
  - batch_size=2, num_workers=0 (for debugging)      
                                                                                                                                                                                                   
  Drop breakpoint() at navdp_trainer.py:80 (start of compute_loss). Then:
                                                                                                                                                                                                   
  cd /home/asus/Research/InternNav
  WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=12345 \                                                                                                                       
    python scripts/train/train.py --name navdp_debug --model-name navdp                                                                                                                            




  ## Dataset verdict — ALL GT available:                                                                                                                                                     
  - Per-frame 4×4 camera pose: parquet['action'] (the loader already reads this as camera_trajectory)                                                                                    
  - Per-frame intrinsic: parquet['observation.camera_intrinsic']                                                                                                                         
  - Per-frame depth (uint16 PNG) → unproject with intrinsic → GT local points                                                                                                            
  - Transform local points by extrinsic → GT world points                                                                                                                                
  - Chassis-to-camera extrinsic T_ext is fixed per-episode (the extrinsic from parquet row 0)    
-                                                                                                                                                                                          
  ## Paper-specified training (Sec V.A, IV.B):                                                                                                                                              
                                                            
  ┌───────┬──────────┬───────┬─────────────────────────────────────────────────────────────────────────┬───────────────────────────┐                                                     
  │ Stage │ Duration │ Batch │                             What's trained                              │       What's frozen       │                                                     
  ├───────┼──────────┼───────┼─────────────────────────────────────────────────────────────────────────┼───────────────────────────┤                                                     
  │ 1     │ 24h      │ 12    │ Geometry decoder + camera_pose_head, local_point_head, world_point_head │ ViT encoder               │                                                     
  ├───────┼──────────┼───────┼─────────────────────────────────────────────────────────────────────────┼───────────────────────────┤                                                     
  │ 2     │ 3 days   │ 32    │ Diffusion head + task-specific heads                                    │ Geometry backbone decoder │                                                     
  └───────┴──────────┴───────┴─────────────────────────────────────────────────────────────────────────┴───────────────────────────┘  


  ## Loss terms (eqs 2, 4, 6, 11):
                                                                                                                                         
  1. Local points (eq 2): L_local = ‖P̂_local - P_local_gt‖ where P_local_gt = D(u,v) · K⁻¹ · [u v 1]ᵀ
  2. Camera pose (eq 4): L_pose = ‖T̂_c - T_c_gt‖ — parametrized as (x, y, θ) on ground plane (3 DoF; code outputs 5-dim so needs a [x, y, z, sinθ, cosθ] decoder)                        
  3. World points (eq 6): L_world = ‖P̂_world - P_world_gt‖ with sign-preserving exp parameterization                                                             
  4. Diffusion (eq 11): standard DDPM ε-prediction on aₜ = (Δxₜ, Δyₜ, Δθₜ), T=24                                                                                                         
  5. Goal / sub-goal — exists in Table III ablation but loss form not written in paper (the pg_pred_mlp head)                                                                            
  6. Critic — NavDP has this; LoGoPlanner paper doesn't explicitly mention it but LoGoPlanner_Policy has a critic_head and cs_pred_mlp → likely retained from NavDP training       
   
  ## What the paper doesn't give us — the unknowns: 

  - Loss weights (λ_local, λ_pose, λ_world, λ_diffusion, λ_goal, λ_critic)  
                                                                                                               
  - Norm type per term (L1/L2/Huber)  
                                                                                                                                                     
  - LR, optimizer, schedule       

## Implementation
  Default loss weights I'd start with: {diffusion: 1.0, critic: 1.0, local: 0.5, world: 0.5, pose: 1.0, subgoal: 0.1} — rationale: diffusion/critic match NavDP defaults, pose is the    
  most important ablation contribution (Table III), points losses at 0.5 since they're dense per-pixel, subgoal small since it's 3-dim regression.  

# Trainer

  Core structure — mirrors NavDP's trainer: same __init__, optimizer, scheduler, dataloader, save_model patterns. Only compute_loss diverges.                                            
  
  Loss composition (paper eqs 2, 4, 6, 11 + NavDP-inherited critic + paper's "Goal" ablation): 
                                                      
  ┌──────────────────────────────────────────┬────────────────────────────────────────────┬────────────────┐                                                                             
  │                   Term                   │                   Source                   │ Default weight │
  ├──────────────────────────────────────────┼────────────────────────────────────────────┼────────────────┤                                                                             
  │ action (diffusion ε-prediction, ng + mg) │ NavDP style, eq 11                         │ 1.0            │
  ├──────────────────────────────────────────┼────────────────────────────────────────────┼────────────────┤
  │ critic                                   │ NavDP (code has critic_head)               │ 1.0            │                                                                             
  ├──────────────────────────────────────────┼────────────────────────────────────────────┼────────────────┤                                                                             
  │ pose                                     │ paper eq 4, ExtrinctHead output            │ 1.0            │                                                                             
  ├──────────────────────────────────────────┼────────────────────────────────────────────┼────────────────┤                                                                             
  │ local                                    │ paper eq 2, local-point head               │ 0.5            │
  ├──────────────────────────────────────────┼────────────────────────────────────────────┼────────────────┤                                                                             
  │ world                                    │ paper eq 6, world-point head               │ 0.5            │
  ├──────────────────────────────────────────┼────────────────────────────────────────────┼────────────────┤                                                                             
  │ subgoal                                  │ paper Table III "Goal" column, pg_pred_mlp │ 0.1            │
  └──────────────────────────────────────────┴────────────────────────────────────────────┴────────────────┘                                                                             
  
  Two-stage training is config-driven, not trainer-driven:                                                                                                                               
  - Stage 1 config should set w_diffusion=0, w_critic=0, w_subgoal=0 + unfreeze geometry in model
  - Stage 2 config should use all weights + freeze geometry in model   
                                                                                                                    
                                                                    
  1. Contract with the future model (forward returns a dict with keys noise_pred_ng/mg, ng_noise, mg_noise, label_critic_pred, augment_critic_pred, camera_poses_pred, local_points_pred, world_points_pred, subgoal_pred). Documented at the top of the file.                                                                        
  2. Contract with the future dataset/collate — 13 batch keys listed in the trainer docstring. That's the next piece to build.                                                              
                                                            
  Unknowns I made choices on (flag for review):

  1. Camera pose GT encoding — code's ExtrinctHead.fc_pose outputs 5-dim but paper says 3 DoF (x, y, θ). I left pose GT as [B, N, P] with P to be decided. Most likely [x, y, z, sinθ, cosθ]. Could also be [x, y, sinθ, cosθ, scale]. We'll finalize when writing the model.                                                   
                                                 
  2. Sub-goal GT — paper mentions "Goal" supervision but doesn't specify the target. Best guess: final waypoint of the trajectory expressed in current frame. Dataset will produce this.
   
  3. Critic — paper doesn't mention it but the checkpoint has critic_head and cs_pred_mlp. Kept the NavDP-style critic loss; set w_critic=0 to disable.   
                                  
  4. MSE everywhere — paper doesn't specify norms, NavDP uses .square().mean() throughout.     
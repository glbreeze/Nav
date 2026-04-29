

 Plan:                                                   
  1. Create internnav/model/basemodel/logoplanner/logoplanner_policy.py with:
    - LoGoPlannerNet (HF PreTrainedModel) + LoGoPlannerModelConfig           
    - Reuses the existing NavDP_RGBD_Backbone for memory encoding   
    - Stubs the GeometryModel with a small Conv/Linear block that produces camera_poses / local_points / world_points / state_token / scene_token of correct shapes (skipping the Pi3 backbone).
                                                                                                                 
  2. Create internnav/configs/model/logoplanner.py + scripts/train/configs/logoplanner.py.  
                                                                   
  3. Wire into scripts/train/train.py + scripts/train/configs/__init__.py.                                                  
                                                                       
  4. Fill in runs.sh with a single-GPU, tiny-batch smoke command pointed at traj_data_navdp.               
                                                                                           
  5. Run + iterate on errors.   


### Input
   
  - batch_pg [B, 3] — final goal in ego frame (x, y, θ)                                                                                                                                
  - batch_memory_rgb/depth [B, M, H, W, *] — short history window (M=8) feeding the memory encoder
  - batch_context_rgb/depth [B, N, Hc, Wc, *] — longer geometry window (N=12) feeding the Pi3-based geometry backbone  
  - batch_labels [B, T, 3] — GT action waypoints (target for diffusion, positive example for critic) 
  - batch_augments [B, T, 3] — perturbed "bad" waypoints (negative example for critic) 

### Policy Model Structure

 1. Encode memory and context
  rgbd_embed = p.rgbd_encoder(mem_rgb, mem_depth)                       # (B, M, D)
  (_, state_token, scene_token), (camera_poses_pred, ...) = p.state_encoder(ctx_rgb, ctx_depth) u
  unify_token = torch.cat([state_token, scene_token], dim=1)            # (B, 2N, D)  
  - rgbd_encoder (NavDP-style): compresses recent RGB-D frames into M per-frame tokens. 
  - state_encoder (LoGoPlanner's geometry head on Pi3): returns 
    - state_token / scene_token — per-context-frame features used to condition the policy.
    - camera_poses_pred / local_points_pred / world_points_pred — auxiliary geometry predictions that are directly supervised (ExtrinctHead + per-pixel depth/world-point decoders from the paper). 

 2. Sub-pointgoal head  
  startgoal_embed = p.start_encoder(pg).unsqueeze(1)     
  state_embed     = p.state_decoder(cat([state_token, startgoal_embed])) 
  subgoal_pred    = p.pg_pred_mlp(state_embed).squeeze(1)                # (B, 3)
  Fuses the goal with state tokens to predict an intermediate sub-goal (localization-grounded). Supervised against batch_gt_subgoal. 

 3. Diffusion noise sampling (two branches) 
  ng_noise, ng_time_embed, ng_noisy_action_embed = self._sample_noise(labels)
  mg_noise, mg_time_embed, mg_noisy_action_embed = self._sample_noise(labels)       
  For each branch: draw random DDPM timesteps, add noise to GT labels, embed the noisy action. The model will be trained to predict the noise (ε-prediction).

   Two conditioning regimes:
  - ng (no-goal): goal slots zeroed → teaches unconditional action generation (classifier-free guidance support).   
  - mg (multi-goal): three copies of state_embed (the sub-goal fusion) → goal-conditioned generation.

 4. Build conditioning 
  cond = cat([time_embed, goal_slots×3, rgbd_embed, unify_token]) + cond_pos_embed(cond)     
  This is the encoder-side "memory" of the transformer decoder: time ⊕ goal ⊕ memory history ⊕ context geometry tokens, with learned positional embedding. 

 5. Transformer decoder → noise prediction  
  act_in   = noisy_action_embed + out_pos_embed(noisy_action_embed)
  out      = p.decoder(tgt=act_in, memory=cond, tgt_mask=p.tgt_mask)  
  noise_pred = p.action_head(layernorm(out)) 
  Both branches share the decoder/heads. tgt_mask makes waypoint prediction causal along the trajectory. Output matches the sampled noise → MSE loss in the trainer. 

 6. Critic head (reuses the same decoder)
   
  label_embed   = p.input_embed(labels).detach()
  augment_embed = p.input_embed(augments).detach() 
  cr_out = p.decoder(tgt=..., memory=ng_cond, memory_mask=p.cond_critic_mask)    
  critic_pred = p.critic_head(cr_out.mean(dim=1))[:, 0]                  # (B,) 
  - Uses the no-goal conditioning (so the critic evaluates trajectories on observations alone).  
  - cond_critic_mask restricts which memory tokens the critic can attend to. 
  - Scalar critic score per sample — trained against batch_label_critic (positive) and batch_augment_critic (negative) with MSE. 
  
 7.  Returned dict 
  Packs everything the trainer needs:   
  - Diffusion: noise_pred_{ng,mg} vs {ng,mg}_noise
  - Critic: label_critic_pred, augment_critic_pred  
  - Geometry auxiliaries (from state_encoder): camera_poses_pred, local_points_pred, world_points_pred
  - Sub-goal: subgoal_pred    
                                                                                                                                                                                            
### Input
| Target           | Aspect          |
|------------------|-----------------|
| 168×308 (H×W)    | 308/168 ≈ 1.83  | 
| 480×270 (W×H)    | 1.78            |       

## 

memory_size = the number of recent RGB-D frames kept as short-term observation history fed to the memory encoder. Default: 8.  
                                                                                                                                                                                                     

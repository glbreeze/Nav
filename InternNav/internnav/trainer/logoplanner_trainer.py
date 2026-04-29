import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from internnav.trainer.base import BaseTrainer


class LoGoPlannerTrainer(BaseTrainer):
    """Trainer for LoGoPlanner (Peng et al., arxiv 2512.19629).

    Paper Sec IV.B / V.A specifies:
      - Loss terms: local points (eq 2), camera pose (eq 4), world points (eq 6),
        diffusion action (eq 11). Paper also mentions Goal (sub-pointgoal) and
        implicitly retains NavDP's critic head — both are included here.
      - Two-stage training: stage 1 fine-tunes the geometry decoder + task-specific
        heads (bs=12, 24h); stage 2 trains the diffusion head with geometry backbone
        frozen (bs=32, 3 days). Stage selection is done by setting freeze flags on
        the model and by zeroing individual loss weights in config.

    Loss weights (paper gives no numeric values). Exposed via config.il.loss with
    getattr defaults:
      w_diffusion (default 1.0), w_critic (1.0), w_pose (1.0),
      w_local (0.5), w_world (0.5), w_subgoal (0.1).

    Expected batch keys from the dataset / collate_fn:
      batch_pg              [B, 3]         goal in ego frame (x, y, yaw)
      batch_memory_rgb      [B, M, H, W, 3]
      batch_memory_depth    [B, H, W, 1]   (last-frame depth; matches LoGoPlanner
                                           inference which uses memory_rgbd[:,-1,:,:,3:4])
      batch_context_rgb     [B, N, H, W, 3]  N = context_size (12 in paper)
      batch_context_depth   [B, N, H, W, 1]
      batch_labels          [B, T, 3]      GT action waypoints (Δx, Δy, Δθ), T = 24
      batch_augments        [B, T, 3]      augmented (negative) actions for critic
      batch_label_critic    [B]            GT critic value for labels
      batch_augment_critic  [B]            GT critic value for augments
      batch_gt_camera_poses [B, N, P]      GT camera pose per context frame (P=5 matches
                                           ExtrinctHead.fc_pose; dataset must encode
                                           [x, y, z, sinθ, cosθ] or agreed equivalent)
      batch_gt_local_points [B, N, H, W, 3]  GT local points = D·K⁻¹·[u v 1]ᵀ
      batch_gt_world_points [B, N, H, W, 3]  GT world points = T_cw · local
      batch_gt_subgoal      [B, 3]         GT sub-pointgoal used by pg_pred_mlp
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.writer = None
        self.start_time = time.time()
        if hasattr(self.model, 'module'):
            self.model_device = self.model.module.device
        else:
            self.model_device = self.model.device
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Rank {rank}] Model device: {self.model_device}")

    def _loss_weights(self):
        w = self.config.il.loss
        return {
            'diffusion': getattr(w, 'w_diffusion', 1.0),
            'critic': getattr(w, 'w_critic', 1.0),
            'pose': getattr(w, 'w_pose', 1.0),
            'local': getattr(w, 'w_local', 0.5),
            'world': getattr(w, 'w_world', 0.5),
            'subgoal': getattr(w, 'w_subgoal', 0.1),
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_device = next(model.parameters()).device

        input_keys = [
            'batch_pg', 'batch_memory_rgb', 'batch_memory_depth',
            'batch_context_rgb', 'batch_context_depth',
            'batch_labels', 'batch_augments',
            'batch_label_critic', 'batch_augment_critic',
            'batch_gt_camera_poses', 'batch_gt_local_points',
            'batch_gt_world_points', 'batch_gt_subgoal',
        ]
        inp = {k: inputs[k].to(model_device, non_blocking=True) for k in input_keys}
        torch.cuda.synchronize(model_device)

        out = model(
            inp['batch_pg'],
            inp['batch_memory_rgb'],
            inp['batch_memory_depth'],
            inp['batch_context_rgb'],
            inp['batch_context_depth'],
            inp['batch_labels'],
            inp['batch_augments'],
        )
        # out: dict returned by LoGoPlannerNet.forward (train mode), see contract above.

        ng_action_loss = (out['noise_pred_ng'] - out['ng_noise']).square().mean()
        mg_action_loss = (out['noise_pred_mg'] - out['mg_noise']).square().mean()
        action_loss = 0.5 * ng_action_loss + 0.5 * mg_action_loss

        critic_loss = (
            (out['label_critic_pred'] - inp['batch_label_critic']).square().mean()
            + (out['augment_critic_pred'] - inp['batch_augment_critic']).square().mean()
        )

        pose_loss = (out['camera_poses_pred'] - inp['batch_gt_camera_poses']).square().mean()
        local_loss = (out['local_points_pred'] - inp['batch_gt_local_points']).square().mean()
        world_loss = (out['world_points_pred'] - inp['batch_gt_world_points']).square().mean()

        subgoal_loss = (out['subgoal_pred'] - inp['batch_gt_subgoal']).square().mean()

        w = self._loss_weights()
        loss = (
            w['diffusion'] * action_loss
            + w['critic'] * critic_loss
            + w['pose'] * pose_loss
            + w['local'] * local_loss
            + w['world'] * world_loss
            + w['subgoal'] * subgoal_loss
        )

        outputs = {
            'loss': loss,
            'ng_action_loss': ng_action_loss,
            'mg_action_loss': mg_action_loss,
            'action_loss': action_loss,
            'critic_loss': critic_loss,
            'pose_loss': pose_loss,
            'local_loss': local_loss,
            'world_loss': world_loss,
            'subgoal_loss': subgoal_loss,
        }
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        try:
            lr = self.config.il.lr
            if rank == 0:
                print(f"[Rank 0] Using learning rate: {lr}")
        except AttributeError:
            lr = 1e-4
            if rank == 0:
                print(f"[Rank 0] Warning: Using default learning rate: {lr}")

        model_for_optim = self.model.module if hasattr(self.model, 'module') else self.model

        trainable = [p for p in model_for_optim.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)

        if rank == 0:
            total_params = sum(p.numel() for p in trainable)
            all_params = sum(p.numel() for p in model_for_optim.parameters())
            print(f"[Rank 0] Optimizer created with {len(optimizer.param_groups)} param groups")
            print(f"[Rank 0] Trainable parameters: {total_params:,} / {all_params:,}")

        return optimizer

    def create_scheduler(self, optimizer, num_training_steps: int):
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.5, total_iters=10000
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        print("\n=== create optimizer and scheduler ===")
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_scheduler(self.optimizer, num_training_steps)
        return self.optimizer, self.lr_scheduler

    def get_train_dataloader(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=1234
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.il.batch_size,
            sampler=sampler,
            num_workers=self.config.il.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.data_collator,
        )

    def save_model(self, output_dir, state_dict=None, **kwargs):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model_to_save.state_dict(), output_dir + "logoplanner.ckpt")
        print(f"Saving model to {output_dir} (is DDP: {hasattr(self.model, 'module')})")

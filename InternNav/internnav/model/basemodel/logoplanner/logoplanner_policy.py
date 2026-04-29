"""HuggingFace wrapper for LoGoPlanner training in InternNav.

The underlying model (``LoGoPlanner_Policy``) lives in
``NavDP/baselines/logoplanner/policy_network.py`` and is NOT duplicated here.
This module imports it via ``sys.path`` and adds:

  - ``LoGoPlannerModelConfig``: mirrors ``NavDPModelConfig`` so the trainer's
    ``from_pretrained(... config=config)`` path works.
  - ``LoGoPlannerNet(PreTrainedModel)``: thin wrapper that owns a single
    ``LoGoPlanner_Policy`` instance as ``self.policy``.
  - A training ``forward()`` that reuses ``self.policy``'s submodules
    (rgbd_encoder, state_encoder, start_encoder, state_decoder, pg_pred_mlp,
    input_embed, decoder, action_head, critic_head, time_emb, noise_scheduler,
    cond_pos_embed, out_pos_embed, layernorm, tgt_mask, cond_critic_mask)
    and returns a dict keyed exactly as ``LoGoPlannerTrainer`` expects.

Note on Pi3 weights: ``GeometryModel`` extends ``Pi3`` which instantiates
``dinov2_vitl14_reg(pretrained=False)`` — the network is structurally fine
with random weights. For a smoke test we do not need the released checkpoint.
"""

import os
import sys

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from internnav.configs.model.base_encoders import ModelCfg
from internnav.configs.trainer.exp import ExpCfg


# --- Make NavDP/baselines/logoplanner importable -------------------------
# Repo layout: <ROOT>/InternNav/ and <ROOT>/NavDP/ are siblings.
_THIS = os.path.dirname(os.path.abspath(__file__))
_INTERNNAV_ROOT = os.path.abspath(os.path.join(_THIS, '../../../..'))
_ROOT = os.path.dirname(_INTERNNAV_ROOT)
_LOGO_DIR = os.path.join(_ROOT, 'NavDP', 'baselines', 'logoplanner')
if _LOGO_DIR not in sys.path:
    sys.path.insert(0, _LOGO_DIR)

from policy_network import LoGoPlanner_Policy  # noqa: E402


class LoGoPlannerModelConfig(PretrainedConfig):
    model_type = 'logoplanner'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)

    @classmethod
    def from_dict(cls, config_dict):
        if 'model_cfg' in config_dict:
            config_dict['model_cfg'] = ExpCfg(**config_dict['model_cfg'])
        return super().from_dict(config_dict)


class LoGoPlannerNet(PreTrainedModel):
    config_class = LoGoPlannerModelConfig

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if hasattr(config, 'model_dump'):
            config = cls.config_class(model_cfg=config)

        model = cls(config)
        model.to(model._device)

        if pretrained_model_name_or_path is None or len(pretrained_model_name_or_path) == 0:
            pass
        elif os.path.isdir(pretrained_model_name_or_path):
            incompatible_keys, _ = model.load_state_dict(
                torch.load(os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin'))
            )
            if len(incompatible_keys) > 0:
                print(f'Incompatible keys: {incompatible_keys}')
        else:
            ckpt = torch.load(pretrained_model_name_or_path, map_location='cpu')
            state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            incompatible_keys, _ = model.load_state_dict(state, strict=False)
            if len(incompatible_keys) > 0:
                print(f'Incompatible keys: {incompatible_keys}')

        return model

    def __init__(self, config: LoGoPlannerModelConfig):
        super().__init__(config)
        if isinstance(config, LoGoPlannerModelConfig):
            self.model_config = ModelCfg(**config.model_cfg['model'])
        else:
            self.model_config = config

        il = self.config.model_cfg['il']
        self._device = torch.device(f"cuda:{config.model_cfg['local_rank']}")
        self.image_size = il['image_size']
        self.memory_size = il['memory_size']
        self.predict_size = il['predict_size']
        self.temporal_depth = il['temporal_depth']
        self.attention_heads = il['heads']
        self.input_channels = il['channels']
        self.token_dim = il['token_dim']
        self.context_size = il.get('context_size', 12)

        self.policy = LoGoPlanner_Policy(
            image_size=self.image_size,
            memory_size=self.memory_size,
            context_size=self.context_size,
            predict_size=self.predict_size,
            temporal_depth=self.temporal_depth,
            heads=self.attention_heads,
            token_dim=self.token_dim,
            channels=self.input_channels,
            device=self._device,
        )

    # Keep ``policy.device`` / tgt_mask / cond_critic_mask consistent with HF's
    # .to() — ``LoGoPlanner_Policy`` stores .device as a plain attribute.
    def to(self, device, *args, **kwargs):
        self = super().to(device, *args, **kwargs)
        self._device = device
        self.policy.device = device
        self.policy.tgt_mask = self.policy.tgt_mask.to(device)
        self.policy.cond_critic_mask = self.policy.cond_critic_mask.to(device)
        return self

    # --------------------------------------------------------------------
    # Training forward
    #
    # Matches the call made by ``LoGoPlannerTrainer.compute_loss``:
    #     out = model(batch_pg, batch_memory_rgb, batch_memory_depth,
    #                 batch_context_rgb, batch_context_depth,
    #                 batch_labels, batch_augments)
    # and returns a dict keyed:
    #     noise_pred_ng, noise_pred_mg, ng_noise, mg_noise,
    #     label_critic_pred, augment_critic_pred,
    #     camera_poses_pred, local_points_pred, world_points_pred,
    #     subgoal_pred
    # --------------------------------------------------------------------
    def _sample_noise(self, action):
        device = action.device
        p = self.policy
        noise = torch.randn(action.shape, device=device)
        timesteps = torch.randint(
            0, p.noise_scheduler.config.num_train_timesteps, (action.shape[0],), device=device
        ).long()
        time_embeds = p.time_emb(timesteps).unsqueeze(1)
        noisy_action = p.noise_scheduler.add_noise(action, noise, timesteps)
        noisy_action_embed = p.input_embed(noisy_action)
        return noise, time_embeds, noisy_action_embed

    def forward(
        self,
        batch_pg,
        batch_memory_rgb,
        batch_memory_depth,
        batch_context_rgb,
        batch_context_depth,
        batch_labels,
        batch_augments,
    ):
        import pdb; pdb.set_trace()
        p = self.policy
        device = next(self.parameters()).device

        pg = batch_pg.to(device, dtype=torch.float32)
        mem_rgb = batch_memory_rgb.to(device, dtype=torch.float32)
        mem_depth = batch_memory_depth.to(device, dtype=torch.float32)
        ctx_rgb = batch_context_rgb.to(device, dtype=torch.float32)
        ctx_depth = batch_context_depth.to(device, dtype=torch.float32)
        labels = batch_labels.to(device, dtype=torch.float32)
        augments = batch_augments.to(device, dtype=torch.float32)

        B = pg.shape[0]
        assert mem_rgb.shape[1] == self.memory_size, (
            f"memory_size mismatch: got {mem_rgb.shape[1]}, expected {self.memory_size}"
        )
        assert ctx_rgb.shape[1] == self.context_size, (
            f"context_size mismatch: got {ctx_rgb.shape[1]}, expected {self.context_size}"
        )

        # --- encode memory + context (real forward paths of LoGoPlanner_Policy)
        rgbd_embed = p.rgbd_encoder(mem_rgb, mem_depth)  # (B, M, D)
        (_, state_token, scene_token), (camera_poses_pred, local_points_pred, world_points_pred) = (
            p.state_encoder(ctx_rgb, ctx_depth)
        )
        unify_token = torch.cat([state_token, scene_token], dim=1)  # (B, 2N, D)

        # sub-pointgoal head (trained against batch_gt_subgoal)
        startgoal_embed = p.start_encoder(pg).unsqueeze(1)  # (B, 1, D)
        state_embed = p.state_decoder(torch.cat([state_token, startgoal_embed], dim=1))  # (B, 1, D)
        subgoal_pred = p.pg_pred_mlp(state_embed).squeeze(1)  # (B, 3)

        # --- diffusion: sample noise for ng and mg branches
        ng_noise, ng_time_embed, ng_noisy_action_embed = self._sample_noise(labels)
        mg_noise, mg_time_embed, mg_noisy_action_embed = self._sample_noise(labels)

        nogoal_embed = torch.zeros_like(startgoal_embed)  # (B, 1, D)

        # --- Build conditioning sequences -----
        def build_cond(time_embed, goal_slots):
            # goal_slots: list of three (B, 1, D) tensors
            cond = torch.cat([time_embed, *goal_slots, rgbd_embed, unify_token], dim=1)
            return cond + p.cond_pos_embed(cond)

        # no-goal branch: goal slots are zero
        ng_cond = build_cond(ng_time_embed, [nogoal_embed, nogoal_embed, nogoal_embed])
        # multi-goal branch: use the sub-pointgoal state_embed in all three slots
        mg_cond = build_cond(mg_time_embed, [state_embed, state_embed, state_embed])

        out_pos_embed_nx = p.out_pos_embed(ng_noisy_action_embed)
        ng_act_in = ng_noisy_action_embed + out_pos_embed_nx
        mg_act_in = mg_noisy_action_embed + out_pos_embed_nx

        ng_out = p.decoder(tgt=ng_act_in, memory=ng_cond, tgt_mask=p.tgt_mask)
        ng_out = p.layernorm(ng_out)
        noise_pred_ng = p.action_head(ng_out)

        mg_out = p.decoder(tgt=mg_act_in, memory=mg_cond, tgt_mask=p.tgt_mask)
        mg_out = p.layernorm(mg_out)
        noise_pred_mg = p.action_head(mg_out)

        # --- critic on GT labels and augments (no-goal cond, masked per cond_critic_mask)
        label_embed = p.input_embed(labels).detach()
        augment_embed = p.input_embed(augments).detach()
        label_act_in = label_embed + out_pos_embed_nx
        augment_act_in = augment_embed + out_pos_embed_nx

        cr_label_out = p.decoder(tgt=label_act_in, memory=ng_cond, memory_mask=p.cond_critic_mask)
        cr_label_out = p.layernorm(cr_label_out)
        label_critic_pred = p.critic_head(cr_label_out.mean(dim=1))[:, 0]

        cr_aug_out = p.decoder(tgt=augment_act_in, memory=ng_cond, memory_mask=p.cond_critic_mask)
        cr_aug_out = p.layernorm(cr_aug_out)
        augment_critic_pred = p.critic_head(cr_aug_out.mean(dim=1))[:, 0]

        return {
            'noise_pred_ng': noise_pred_ng,
            'noise_pred_mg': noise_pred_mg,
            'ng_noise': ng_noise,
            'mg_noise': mg_noise,
            'label_critic_pred': label_critic_pred,
            'augment_critic_pred': augment_critic_pred,
            'camera_poses_pred': camera_poses_pred,
            'local_points_pred': local_points_pred,
            'world_points_pred': world_points_pred,
            'subgoal_pred': subgoal_pred,
        }

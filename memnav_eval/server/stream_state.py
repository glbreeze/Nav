"""Incremental LingBot KV + DINO CLS capture for MemNav Isaac eval."""

import numpy as np
import torch

from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream


class LingBotEpisodeState:
    """Stream RGB frames online; build the cache dict expected by LingBotStream.window_forward."""

    def __init__(self, lingbot, device="cuda:0"):
        self.lingbot = lingbot
        self.device = device
        self.model = lingbot.model
        self.agg = self.model.aggregator
        self.num_scale = lingbot.num_scale
        self.psi = lingbot.num_special
        self.L = self.agg.depth
        self.ch = self.model.camera_head
        self.NI = self.ch.num_iterations
        self.TD = self.ch.trunk_depth
        self.reset()

    def reset(self):
        self.frames = []
        self.dino_cls = []
        self.scale_k = self.scale_v = None
        self.anchor_k_list = []
        self.anchor_v_list = []
        self.cam_k_list = []
        self.cam_v_list = []
        # per-frame camera pose (absT, quatR, FoV). The 60a301a policy reads cur_pose from
        # cache["cam_pose_enc"][k] instead of recomputing it via window_forward (cold-start error
        # ATE 3.35m vs 0.04m). The camera head ALREADY produces this pose every frame -- append_frame
        # calls self.ch(...) but discards its return; we now keep pl[-1][0], byte-for-byte the same
        # value precompute stores (identical causal_inference calls on the same continuous stream).
        self.cam_pose_list = []
        self.model.clean_kv_cache()
        self.ch.clean_kv_cache()

    def _read_cache_full(self, frame_slice):
        kv = self.agg.kv_cache
        ks = torch.stack([kv[f"k_{i}"][0, :, frame_slice].to(torch.float16).cpu() for i in range(self.L)])
        vs = torch.stack([kv[f"v_{i}"][0, :, frame_slice].to(torch.float16).cpu() for i in range(self.L)])
        return ks, vs

    def _read_cache_anchor_newest(self):
        kv = self.agg.kv_cache
        n = self.psi
        ks = torch.stack([kv[f"k_{i}"][0, :, -1, :n].to(torch.float16).cpu() for i in range(self.L)])
        vs = torch.stack([kv[f"v_{i}"][0, :, -1, :n].to(torch.float16).cpu() for i in range(self.L)])
        return ks, vs

    def _read_cam_newest(self, n_new):
        cc = self.ch.kv_cache
        ks = torch.stack([torch.stack([cc[it][f"k_{bl}"][0, :, -n_new:, 0].to(torch.float16).cpu()
                                       for bl in range(self.TD)]) for it in range(self.NI)])
        vs = torch.stack([torch.stack([cc[it][f"v_{bl}"][0, :, -n_new:, 0].to(torch.float16).cpu()
                                       for bl in range(self.TD)]) for it in range(self.NI)])
        return ks.permute(3, 0, 1, 2, 4).contiguous(), vs.permute(3, 0, 1, 2, 4).contiguous()

    @torch.no_grad()
    def append_frame(self, img_chw):
        """img_chw: [3,H,W] float tensor on device (LingBot-preprocessed)."""
        self.frames.append(img_chw)
        k = len(self.frames) - 1
        dev = self.device
        scale = self.num_scale

        if k + 1 < scale:
            cls = self.lingbot.dino(img_chw.unsqueeze(0))["cls"][0]
            self.dino_cls.append(cls.cpu())
            return

        if k + 1 == scale:
            block = torch.stack(self.frames[:scale], 0).unsqueeze(0).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scale_agg, _psi = self.model._aggregate_features(
                    block, num_frame_for_scale=scale, num_frame_per_block=scale)
                pl = self.ch(scale_agg, causal_inference=True, num_frame_per_block=scale,
                             num_frame_for_scale=scale)
            self.cam_pose_list.append(pl[-1][0].float().cpu())      # [scale, 9]
            ck, cv = self._read_cam_newest(scale)
            self.cam_k_list.append(ck)
            self.cam_v_list.append(cv)
            self.scale_k, self.scale_v = self._read_cache_full(slice(0, scale))
            cls_block = self.lingbot.dino(block[0])["cls"]
            self.dino_cls = [cls_block[i].cpu() for i in range(scale)]
            return

        frame = img_chw.unsqueeze(0).unsqueeze(0).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            agg_tok, psi = self.model._aggregate_features(
                frame, num_frame_for_scale=scale, num_frame_per_block=1)
            pl = self.ch(agg_tok, causal_inference=True, num_frame_per_block=1, num_frame_for_scale=scale)
        # Keep the tokens instead of discarding them: this frame was just aggregated against
        # the correct causal cache, which is exactly what window_forward spends 32 frames
        # reconstructing, and encode_memory only ever uses this last row.
        self.cur_agg = [t for t in agg_tok]
        self.psi_idx = psi
        self.cam_pose_list.append(pl[-1][0].float().cpu())          # [1, 9]
        cls = self.lingbot.dino(img_chw.unsqueeze(0))["cls"][0]
        self.dino_cls.append(cls.cpu())
        ak, av = self._read_cache_anchor_newest()
        self.anchor_k_list.append(ak)
        self.anchor_v_list.append(av)
        ck, cv = self._read_cam_newest(1)
        self.cam_k_list.append(ck)
        self.cam_v_list.append(cv)

    def ready(self):
        return self.scale_k is not None

    def snapshot_kv(self):
        """Clone the live KV cache so it can be put back byte-for-byte.

        Needed because goal_append_warm starts with _inject -> clean_kv_cache() and leaves the
        cache holding the goal warm-up rather than the stream. Rebuilding by replaying the
        window (the obvious alternative) does NOT work: that reconstruction sits ~0.5 m away
        from the live stream, so repairing with it compounds the error instead of removing it
        (measured: 0.6250 m drift before, 1.2993 m after, growing). The live tensors are the
        correct state -- keep them.
        """
        agg = self.agg
        ch = self.ch
        snap = {
            "agg": {k: (v.clone() if torch.is_tensor(v) else v) for k, v in agg.kv_cache.items()},
            "agg_total": int(getattr(agg, "total_frames_processed", 0)),
            "ch": [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()}
                   for d in (ch.kv_cache or [])],
            "ch_idx": int(getattr(ch, "frame_idx", 0)),
        }
        if not getattr(self, "_snap_logged", False):
            nb = sum(v.numel() * v.element_size()
                     for v in snap["agg"].values() if torch.is_tensor(v))
            nb += sum(v.numel() * v.element_size()
                      for d in snap["ch"] for v in d.values() if torch.is_tensor(v))
            print(f"[stream] KV snapshot = {nb / 1e9:.2f} GB", flush=True)
            self._snap_logged = True
        return snap

    def restore_kv(self, snap):
        """Put back a snapshot_kv() result, exactly."""
        self.agg.kv_cache = snap["agg"]
        self.agg.total_frames_processed = snap["agg_total"]
        self.ch.kv_cache = snap["ch"]
        self.ch.frame_idx = snap["ch_idx"]

    def build_cache(self):
        """Device cache dict for window_forward / camera_pose."""
        if not self.ready():
            raise RuntimeError("need at least num_scale frames before inference")
        if self.anchor_k_list:
            anchor_k = torch.stack(self.anchor_k_list, 0).numpy()
            anchor_v = torch.stack(self.anchor_v_list, 0).numpy()
        else:
            Hh, d = self.cam_k_list[0].shape[3], self.cam_k_list[0].shape[-1]
            anchor_k = np.zeros((0, self.L, Hh, self.psi, d), np.float16)
            anchor_v = np.zeros((0, self.L, Hh, self.psi, d), np.float16)
        cam_k = torch.cat(self.cam_k_list, 0).numpy()
        cam_v = torch.cat(self.cam_v_list, 0).numpy()
        sk, sv, ak, av = LingBotStream._cache_to_layered(
            self.scale_k.numpy(), self.scale_v.numpy(), anchor_k, anchor_v, self.device)
        ck, cv = LingBotStream._cam_to_device(cam_k, cam_v, self.device)
        # cam_pose_enc [S,9] on device: same key the disk cache exposes, so the policy's
        # cur_pose = cache["cam_pose_enc"][k] path works identically online.
        cam_pose_enc = torch.cat(self.cam_pose_list, 0).to(self.device, torch.float32)
        return dict(scale_k=sk, scale_v=sv, anchor_k=ak, anchor_v=av, cam_k=ck, cam_v=cv,
                    cam_pose_enc=cam_pose_enc,
                    # tokens for the newest frame, already computed by append_frame against
                    # the live causal cache -- lets encode_memory skip the window replay.
                    cur_agg=getattr(self, "cur_agg", None),
                    psi=getattr(self, "psi_idx", None))

    def warm_frames(self, m, warm):
        """The preprocessed frames [max(num_scale, m-warm+1) .. m] the goal warm-up needs.
        goal_append_warm normally re-reads these from rgb_dir/*.jpg; online we already hold the
        exact same preprocessed tensors in self.frames, so hand them over and skip disk +
        double-preprocessing. Range MUST match goal_append_warm_frames' internal start."""
        m = max(0, min(m, len(self.frames) - 1))               # valid frame index
        start = min(max(self.num_scale, m - warm + 1), m)      # floor start<=m -> never empty
        return torch.stack(self.frames[start:m + 1], 0)   # [m-start+1, 3, H, W]

    def mem_cls_tensor(self):
        return torch.stack(self.dino_cls, 0)

    def window_tensor(self, k, window):
        """Last `window` frames ending at k (pad start if needed). [W,3,H,W]."""
        W = window
        start = max(0, k - W + 1)
        frames = self.frames[start:k + 1]
        if len(frames) < W:
            pad = [frames[0]] * (W - len(frames))
            frames = pad + frames
        return torch.stack(frames[-W:], 0)

    def match_window_tensor(self, m, window):
        start = max(0, m - window + 1)
        frames = self.frames[start:m + 1]
        if len(frames) < window:
            pad = [frames[0]] * (window - len(frames))
            frames = pad + frames
        return torch.stack(frames[-window:], 0)

#!/usr/bin/env python
"""Revisit vs novel image-goal eval in Habitat MP3D.

WHY THIS EXISTS
---------------
eval_imagegoal_habitat.py samples a random (start, goal) pair, so the goal is a place the
robot has never been: the gate says "novel", the revisit tokens get masked out, and the
memory is never consulted. Every number it produces measures the novel branch alone --
which is the part MemNav shares with a plain image-goal policy. The retrieval work (covis
labels, the decoupled ranking loss, E(k)) is invisible to it.

This driver mirrors how the TRAINING data is built, so the eval distribution matches the one
the checkpoint was fit on. generate_twoleg.py makes a 2-leg episode as:

    leg 1   start ----> A     GT path, streamed into memory. The robot walks PAST the goal.
    leg 2   A     ----> B     the policy drives, given B's image.   <-- the measured part

where B comes from `sample_revisit`: pick a frame X the robot actually occupied on leg 1,
jitter the goal inside a 1.5 m disk around X, take X's heading +/- 45 deg, and require
co-visibility with the history in [0.20, 1.00]. There is no third leg -- "walk away" IS the
rest of leg 1.

Both conditions replay the identical rendered leg-1 frames and start from the same pose (A):

      revisit run: goal from G.sample_revisit  (seen on leg 1, and inside E(k))
      novel   run: goal from G.sample_novel    (max covisibility with leg 1 < 0.10)

so SR_revisit - SR_novel is the contribution of retrieval. Both samplers are CALLED, not
reimplemented, so acceptance is bit-for-bit the training rule.

GEOMETRY -- and the trap this replaces
--------------------------------------
E(k) = [anchor_margin .. k - exclude_recent] = [39 .. k-83]. With the goal drawn around
anchor frame X and the measured leg starting at k = len(leg1)-1, membership is simply

    39 <= X <= len(leg1) - 1 - 83

i.e. a constraint on the FRAME INDEX. An earlier version of this file instead manufactured
the separation with distance, forcing geodesic >= 83 * 0.0376 * 1.5 = 4.68 m between the goal
and the start of the measured leg. Geodesic distance is symmetric, so that also forced the
navigation itself to >= 4.68 m -- beyond anything the policy has solved (its successes top out
near 2.3 m). Every revisit episode failed while match_hit sat at a perfect 4/4, and capping
the return at 3.0 m sampled zero episodes, ">= 4.68 and <= 3.0" being empty. Frame separation
and physical distance are different quantities; on a curving path a goal 100 frames back can
be 2 m away. Keep them separate: X bounds the frames, MEMNAV_DBACK_HI bounds the metres.

Legs use pursuit_track (the controller that generated the training trajectories), so streamed
frames carry train-time spacing and "83 frames" means here what it means in the loader.

METRICS
-------
SR is downstream of MPC, collision sliding and trajectory choice, so it can hide a working
retrieval (or flatter a broken one). Two direct measurements are recorded alongside it:

    gate        P(revisit) from the model. Want high on revisit runs, low on novel runs.
    match_hit   whether the retrieved frame is actually near X. Ground truth, since the
                sampler returns X's index -- not an estimate.

Success is position-only: |pos - goal_xz| < success_dist. No heading term.

Usage:
  python eval_revisit_habitat.py --scene X.glb --port 28890 --num_episodes 20
"""
import argparse
import io
import json
import os
import sys
import time

from types import SimpleNamespace

import numpy as np
import requests
from PIL import Image, ImageDraw

MEMNAVDATA = os.environ.get("MEMNAVDATA_DIR", "/scratch/ay2710/Nav_memnav/MemNavData")
if MEMNAVDATA not in sys.path:
    sys.path.insert(0, MEMNAVDATA)
import generate_twoleg as G  # noqa: E402  (make_sim, render, geodesic, densify, elastic_smooth,
                             #              build_esdf, pursuit_track, K)
from mpc_tracking import MPC_Controller  # noqa: E402

DT = 0.1
FWD_SIGN = float(os.environ.get("MEMNAV_FWD_SIGN", "1"))
LAT_SIGN = float(os.environ.get("MEMNAV_LAT_SIGN", "1"))
# default 1: the habitat client's yaw->axis mapping is transposed; with 0 the robot spins in
# place and every episode scores 0. See memnav_habitat_eval_progress.md.
SWAP_XY = os.environ.get("MEMNAV_SWAP_XY", "1") == "1"

ANCHOR_MARGIN = int(os.environ.get("MEMNAV_ANCHOR_MARGIN", "39"))   # num_scale + window - 1
EXCLUDE_RECENT = int(os.environ.get("MEMNAV_EXCLUDE_RECENT", "83"))
V_PER_FRAME = 0.0376   # pursuit_track's per-frame advance (train-time controller)


# --------------------------------------------------------------------------- #
# server IO
# --------------------------------------------------------------------------- #
def _jpg(rgb):
    b = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(b, format="JPEG", quality=95)
    return b.getvalue()


def _png_depth(depth_m):
    d = np.clip(depth_m * 10000.0, 0, 65535).astype(np.uint32)
    b = io.BytesIO()
    Image.fromarray(d.squeeze().astype(np.int32), mode="I").save(b, format="PNG")
    return b.getvalue()


def navigator_reset(port, stop_threshold):
    requests.post(f"http://localhost:{port}/navigator_reset",
                  json={"intrinsic": G.K.tolist(), "stop_threshold": stop_threshold,
                        "batch_size": 1}).raise_for_status()


def stream_frame(rgb, port):
    """Memory-only append (legs 1-2). No goal needed, no policy run."""
    r = requests.post(f"http://localhost:{port}/stream_frame",
                      files={"image": ("i.jpg", _jpg(rgb), "image/jpeg")})
    r.raise_for_status()
    return int(json.loads(r.text)["k"])


def set_goal(goal_rgb, port):
    """Install the goal after the memory is built. Unlike navigator_reset_env this keeps
    the stream state."""
    r = requests.post(f"http://localhost:{port}/set_goal",
                      files={"goal": ("g.jpg", _jpg(goal_rgb), "image/jpeg")})
    r.raise_for_status()


def imagegoal_step(goal_rgb, rgb, depth, port):
    r = requests.post(f"http://localhost:{port}/imagegoal_step",
                      files={"image": ("i.jpg", _jpg(rgb), "image/jpeg"),
                             "goal": ("g.jpg", _jpg(goal_rgb), "image/jpeg"),
                             "depth": ("d.png", _png_depth(depth), "image/png")},
                      data={"depth_time": time.time(), "rgb_time": time.time()})
    j = json.loads(r.text)
    if "error" in j:
        raise RuntimeError(j["error"])
    gate = j.get("revisit_gate", [None])[0]
    match = j.get("match_idx", [None])[0]
    return np.array(j["trajectory"]), gate, match


# --------------------------------------------------------------------------- #
# geometry (must match eval_imagegoal_habitat.py exactly)
# --------------------------------------------------------------------------- #
def fwd_xz(yaw):  return np.array([-np.sin(yaw), -np.cos(yaw)])
def left_xz(yaw): return np.array([-np.cos(yaw),  np.sin(yaw)])


def _geo(pf, a, b):
    ok, gd, pts = G.geodesic(pf, a, b)
    return (ok and np.isfinite(gd)), float(gd) if ok else np.inf, pts


def generator_args(cam_h):
    """The generator defaults that govern goal sampling, verbatim from
    generate_twoleg.py's argparse. Passed straight into its sampling functions so the eval
    accepts exactly the goals the training data accepted."""
    return SimpleNamespace(
        anchor_margin=ANCHOR_MARGIN,      # 39  = num_scale + window - 1
        min_recall_gap=EXCLUDE_RECENT,    # 83  = covis onset horizon
        long_term_frac=0.7,               # P(force the anchor outside the current window)
        goal_jitter_pos=1.50,             # uniform disk radius around the anchor frame (m)
        head_max_deg=45.0,                # heading cone around the anchor frame
        covis_lo=0.20, covis_hi=1.00,     # revisit: max-covisibility band
        novel_covis=0.10,                 # novel: max covisibility must stay under this
        # generate_twoleg defaults this to 3.0, but that is a DATA-generation choice, not a
        # comparison one: it puts every novel goal >= 3 m out while revisit goals sit under
        # MEMNAV_DBACK_HI, so the two conditions never overlap in distance and SR_revisit
        # beats SR_novel for reasons that have nothing to do with memory. The eval is a
        # controlled comparison, so the floor is dropped here and distance is matched to the
        # revisit goal explicitly below.
        min_dist_AB=float(os.environ.get("MEMNAV_NOVEL_DMIN", "0.5")),
        covis_stride=4, covis_tol=0.30,
        r_min=0.40, floor_tol=0.80,
        goal_tries=40, cam_h=cam_h,
    )


def build_episode(sim, pf, esdf, rng, args, gargs, tries=400):
    """One episode: walk `start -> A`, then draw a revisit goal and a novel goal against that
    leg using the generator's own samplers.

    Returned frames are rendered ONCE here and replayed per condition, so both conditions see
    a byte-identical memory and start from the same pose -- the comparison isolates "was the
    goal seen" and nothing else.
    """
    ch = np.array([0.0, gargs.cam_h, 0.0])
    d_lo = float(os.environ.get("MEMNAV_DA_MIN", "5.0"))
    d_hi = float(os.environ.get("MEMNAV_DA_MAX", "9.0"))
    back_hi = float(os.environ.get("MEMNAV_DBACK_HI", "3.0"))

    for _ in range(tries):
        a = np.array(pf.get_random_navigable_point())
        s = np.array(pf.get_random_navigable_point())
        floor_y = float(a[1])
        if abs(s[1] - a[1]) > gargs.floor_tol:
            continue
        ok, g1, pts = _geo(pf, s, a)
        if not ok or not (d_lo <= g1 <= d_hi):
            continue

        ref = G.elastic_smooth(G.densify(pts, 0.20), pf, esdf)
        leg1, _done = G.pursuit_track(ref, pf, init_pos=s, init_theta=None,
                                      cam_h=gargs.cam_h, floor_y=floor_y)
        # the anchor must sit >= EXCLUDE_RECENT frames back AND >= ANCHOR_MARGIN in, so the
        # leg has to be long enough to hold both margins at once.
        if not leg1 or len(leg1) < ANCHOR_MARGIN + EXCLUDE_RECENT + 2:
            continue

        poses, depths, rgbs = [], [], []
        for cam_pos, psi in leg1:
            rgb, dep = G.render(sim, cam_pos, psi)
            rgbs.append(rgb)
            depths.append(dep)
            poses.append(G.cam_to_world_hab(cam_pos, psi))

        rv = G.sample_revisit(sim, pf, leg1, poses, depths, len(leg1), rng, gargs, ch, floor_y)
        if rv is None:
            continue
        c_pos, c_yaw, c_cov, anchor_i, head_off, _curve = rv
        # E(k) membership at the moment leg 2 starts: k = len(leg1) - 1.
        if not (ANCHOR_MARGIN <= anchor_i <= len(leg1) - 1 - EXCLUDE_RECENT):
            continue
        okc, geo_back, _ = _geo(pf, a, c_pos)
        if not okc or geo_back > back_hi:
            continue

        # DISTANCE-MATCHED novel control. Without this the comparison is confounded: an
        # unmatched novel goal averaged 6.77 m against revisit's 2.47 m (dry run 14260534),
        # so any SR gap would just be measuring "near vs far", not "remembered vs not".
        tol = float(os.environ.get("MEMNAV_MATCH_TOL_M", "0.5"))
        n_pos = n_yaw = n_cov = None
        geo_novel = np.inf
        for _ in range(int(os.environ.get("MEMNAV_NOVEL_TRIES", "40"))):
            nv = G.sample_novel(sim, pf, a, poses, depths, rng, gargs, ch, floor_y)
            if nv is None:
                continue
            cand_p, cand_yaw, cand_cov, _ncurve = nv
            okn, gn, _ = _geo(pf, a, cand_p)
            if okn and abs(gn - geo_back) <= tol:
                n_pos, n_yaw, n_cov, geo_novel = cand_p, cand_yaw, cand_cov, gn
                break
        if n_pos is None:
            continue

        return dict(start=s, a=a, floor_y=floor_y, leg1=leg1, rgbs=rgbs,
                    c_pos=c_pos, c_yaw=float(c_yaw), c_cov=float(c_cov),
                    anchor_i=int(anchor_i), head_off=float(head_off), geo_back=float(geo_back),
                    n_pos=n_pos, n_yaw=float(n_yaw), n_cov=float(n_cov),
                    geo_novel=float(geo_novel), leg1_m=float(g1), leg1_frames=len(leg1),
                    frame_gap=int(len(leg1) - 1 - anchor_i))
    return None


def goal_view_yaw(pf, frm, to):
    """Yaw of the view at `to` when arriving from `frm`, i.e. what the robot saw when it
    walked past. A random yaw could face a wall and be unlocalizable, dooming any model."""
    ok, _gd, pts = G.geodesic(pf, frm, to)
    if ok and pts is not None and len(pts) >= 2:
        seg = np.asarray(pts[-1]) - np.asarray(pts[-2])
    else:
        seg = np.asarray(to) - np.asarray(frm)
    return float(np.arctan2(-seg[0], -seg[2]))


# --------------------------------------------------------------------------- #
# legs
# --------------------------------------------------------------------------- #
def _panel(rgb, label):
    """One labelled video panel. cv2 is not imported here; PIL already is."""
    im = Image.fromarray(np.asarray(rgb).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14], fill=(0, 0, 0))
    d.text((3, 2), label, fill=(255, 255, 0))
    return np.asarray(im)


def _overlay(rgb, goal_rgb, mem_rgb, d_goal, step, gate, match):
    """[ what it sees | what it was asked to find | what it retrieved ].

    The third panel is the revisit-specific part: match_idx comes back from the server, so
    the frame the model actually matched can sit next to the goal it was given. If those two
    panels show different places, retrieval is wrong -- visible at a glance, whereas
    match_hit only says so numerically and only when ground truth exists.
    """
    g = "n/a" if gate is None else f"{float(gate):.2f}"
    panels = [_panel(rgb, f"s{step} d={d_goal:.2f}"), _panel(goal_rgb, "GOAL")]
    if mem_rgb is not None:
        panels.append(_panel(mem_rgb, f"matched #{match} gate={g}"))
    h = min(p.shape[0] for p in panels)
    return np.concatenate([p[:h] for p in panels], axis=1)


def drive_gt_leg(sim, pf, esdf, a, b, init_pos, init_theta, cam_h, floor_y, port,
                 rgb_sink=None):
    """Walk a->b along the GT path, streaming every frame into memory.

    Uses pursuit_track -- the controller that generated the training trajectories -- rather
    than the eval MPC, so the streamed frames carry train-time spacing (0.0376 m/frame).
    Returns the frame list, or None if the path is unusable.

    `rgb_sink`, when given, collects the rendered frames in stream order so the video can
    show the one the model retrieves (match_idx indexes this exact sequence).
    """
    ok, _gd, pts = G.geodesic(pf, a, b)
    if not ok or pts is None or len(pts) < 2:
        return None
    ref = G.elastic_smooth(G.densify(pts, 0.20), pf, esdf)
    frames, _done = G.pursuit_track(ref, pf, init_pos=init_pos, init_theta=init_theta,
                                    cam_h=cam_h, floor_y=floor_y)
    if not frames:
        return None
    for cam_pos, psi in frames:
        rgb, _depth = G.render(sim, cam_pos, psi)
        if rgb_sink is not None:
            rgb_sink.append(rgb)
        stream_frame(rgb, port)
    return frames


def run_policy_leg(sim, pf, start_pos_xz, start_yaw, goal_rgb, goal_xz,
                   c_frame_idx, floor_y, port, args, mem_rgbs=None):
    """Leg 3: the policy drives to the goal. Same MPC/unicycle loop as the plain image-goal
    eval, so the two evals' SRs are comparable. Records gate/match alongside SR.

    Returns (metrics, frames) -- frames is the overlay video, empty unless --save_video.
    """
    set_goal(goal_rgb, port)
    pos = np.array(start_pos_xz, dtype=float)
    yaw = float(start_yaw)
    path_len, stop_run = 0.0, 0
    gates, match_hits = [], []
    frames = []
    srv_time, srv_n = 0.0, 0
    reached = False
    step = 0
    for step in range(args.max_steps):
        cam_pos = np.array([pos[0], floor_y + args.cam_h, pos[1]])
        rgb, depth = G.render(sim, cam_pos, yaw)
        d_goal = float(np.linalg.norm(pos - goal_xz))
        if step % 25 == 0:
            gv = goal_xz - pos
            brg = np.degrees(np.arctan2(float(gv @ left_xz(yaw)), float(gv @ fwd_xz(yaw))))
            avg_ms = 1000.0 * srv_time / max(srv_n, 1)
            g_last = f"{gates[-1]:.2f}" if gates else "n/a"
            print(f"    step {step:3d} d_goal={d_goal:.2f} goal_bearing={brg:.0f}deg "
                  f"path={path_len:.1f} gate={g_last} srv={avg_ms:.0f}ms/step", flush=True)
        if d_goal < args.success_dist:
            reached = True
            break
        _t0 = time.time()
        traj, gate, match = imagegoal_step(goal_rgb, rgb, depth, port)
        srv_time += time.time() - _t0
        srv_n += 1
        pts3 = np.array(traj).reshape(-1, 3)
        is_warmup = np.allclose(pts3, np.array([0.0, 0.1, 0.0]), atol=1e-3)
        if gate is not None:
            gates.append(float(gate))
            # C is not a single frame: the robot was near it for a stretch. Count a hit if the
            # retrieved frame is within tol frames of C's index -- those all show the place.
            if match is not None and c_frame_idx is not None:
                match_hits.append(int(abs(int(match) - c_frame_idx) <= args.match_tol))
        # Right after warmup the robot has barely moved, so the goal bearing is still close to
        # the start bearing -- the cleanest moment to ask whether the MODEL points at the goal
        # at all. Separates "aims right, cannot close the distance" from "aims at nothing",
        # which SR and final_dist cannot distinguish. Same probe as the image-goal eval.
        if 15 <= step <= 22 and not is_warmup and pts3.size:
            gv = goal_xz - pos
            true_brg = np.degrees(np.arctan2(float(gv @ left_xz(yaw)), float(gv @ fwd_xz(yaw))))
            # Apply the SAME axis convention the driving code below uses. Reading pts3 raw
            # (index0=fwd, index1=lat) is only correct when SWAP_XY=0; with SWAP_XY=1 -- the
            # default, and what every run so far used -- the driver swaps the two, so a raw
            # reading reports a bearing the robot never steers to. That made "novel agrees"
            # look true when it was not; revisit was far enough off that it stayed wrong
            # either way.
            _mf, _ml = FWD_SIGN * float(pts3[-1, 0]), LAT_SIGN * float(pts3[-1, 1])
            mfx, mlat = (_ml, _mf) if SWAP_XY else (_mf, _ml)
            model_brg = np.degrees(np.arctan2(mlat, mfx))
            print(f"    [diag s{step}] MODEL fwd={mfx:.2f} lat={mlat:.2f} dir={model_brg:.0f}deg "
                  f"vs TRUE={true_brg:.0f}deg (agree? {abs(model_brg - true_brg) < 45}) "
                  f"gate={gate} match={match}", flush=True)
        if args.save_video and step % args.vid_stride == 0:
            mem_rgb = None
            if mem_rgbs and match is not None and 0 <= int(match) < len(mem_rgbs):
                mem_rgb = mem_rgbs[int(match)]
            frames.append(_overlay(rgb, goal_rgb, mem_rgb, d_goal, step, gate, match))
        if is_warmup:
            v_cmd, w_cmd = args.v_max, 0.0
        else:
            if pts3.size == 0 or np.abs(pts3).max() < 1e-3:
                stop_run += 1
                if stop_run >= args.stop_patience:
                    break
                continue
            stop_run = 0
            fx = FWD_SIGN * pts3[:, 0]
            lat = LAT_SIGN * pts3[:, 1]
            if SWAP_XY:
                fx, lat = lat.copy(), fx.copy()
            fwd, lft = fwd_xz(yaw), left_xz(yaw)
            W = pos[None, :] + fx[:, None] * fwd[None, :] + lat[:, None] * lft[None, :]
            W = np.vstack([pos[None, :], W])
            theta = float(np.arctan2(-np.cos(yaw), -np.sin(yaw)))
            try:
                mpc = MPC_Controller(W, N=15, desired_v=args.v_max, v_max=args.v_max,
                                     w_max=args.w_max, ref_gap=3)
                u, _ = mpc.solve(np.array([pos[0], pos[1], theta]))
                v_cmd, w_cmd = float(u[0, 0]), float(u[0, 1])
            except Exception:
                v_cmd, w_cmd = 0.0, 0.0
        proposed = pos + v_cmd * DT * fwd_xz(yaw)
        yaw = yaw - w_cmd * DT
        a3 = np.array([pos[0], floor_y, pos[1]])
        b3 = np.array([proposed[0], floor_y, proposed[1]])
        try:
            nxt = np.array(pf.try_step(a3, b3))
        except Exception:
            snap = np.array(pf.snap_point(b3))
            nxt = snap if np.linalg.norm(snap[[0, 2]] - b3[[0, 2]]) < args.v_max * DT * 2 else a3
        new_pos = np.array([nxt[0], nxt[2]])
        path_len += float(np.linalg.norm(new_pos - pos))
        pos = new_pos

    d_final = float(np.linalg.norm(pos - goal_xz))
    reached = reached or (d_final < args.success_dist)
    return dict(success=int(reached), final_dist=d_final, path_len=path_len, steps=step + 1,
                gate_mean=float(np.mean(gates)) if gates else None,
                match_hit=float(np.mean(match_hits)) if match_hits else None), frames


def run_condition(sim, pf, ep, cond, port, args):
    """Replay leg 1 into a fresh memory, then navigate from A to this condition's goal.

    Both conditions replay the SAME rendered frames, so the memory and the start pose are
    identical and only the goal differs.
    """
    navigator_reset(port, args.stop_threshold)          # fresh memory per condition
    mem_rgbs = []
    for rgb in ep["rgbs"]:
        mem_rgbs.append(rgb)
        stream_frame(rgb, port)

    if cond == "revisit":
        goal_pos, goal_yaw, gt_idx = ep["c_pos"], ep["c_yaw"], ep["anchor_i"]
    else:
        goal_pos, goal_yaw, gt_idx = ep["n_pos"], ep["n_yaw"], None
    goal_rgb, _ = G.render(
        sim, np.array([goal_pos[0], goal_pos[1] + args.cam_h, goal_pos[2]]), goal_yaw)

    last_pos, last_psi = ep["leg1"][-1]
    start_xz = np.array([last_pos[0], last_pos[2]])
    goal_xz = np.array([goal_pos[0], goal_pos[2]])

    r, frames = run_policy_leg(sim, pf, start_xz, last_psi, goal_rgb, goal_xz,
                               gt_idx, ep["floor_y"], port, args, mem_rgbs=mem_rgbs)
    k = len(ep["leg1"]) - 1
    r.update(leg1_frames=ep["leg1_frames"], c_frame_idx=ep["anchor_i"], k_at_leg3=k,
             frame_gap=ep["frame_gap"], covis=ep["c_cov"] if cond == "revisit" else ep["n_cov"],
             geo=ep["geo_back"] if cond == "revisit" else ep["geo_novel"],
             c_in_ek=int(ANCHOR_MARGIN <= ep["anchor_i"] <= k - EXCLUDE_RECENT))
    return r, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--navmesh", default="")
    ap.add_argument("--port", type=int, default=28890)
    ap.add_argument("--num_episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--success_dist", type=float, default=1.0)
    ap.add_argument("--v_max", type=float, default=0.4)
    ap.add_argument("--w_max", type=float, default=1.2)
    ap.add_argument("--cam_h", type=float, default=0.5)
    ap.add_argument("--stop_threshold", type=float, default=-1000.0)
    ap.add_argument("--stop_patience", type=int, default=8)
    ap.add_argument("--match_tol", type=int, default=25, help="frames around C counted as a hit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="revisit_out")
    ap.add_argument("--dry_run", action="store_true",
                    help="sample episodes and print the distribution; no server, no policy")
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--vid_stride", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sim = G.make_sim(args.scene, args.navmesh or None)
    pf = sim.pathfinder

    floors = G.detect_floors(pf)
    # detect_floors returns [(floor_y, n_samples), ...] sorted by sample count, so the
    # dominant floor's height is floors[0][0] -- not floors[0] (that is the whole tuple).
    esdf = G.build_esdf(pf, float(floors[0][0]) if len(floors) else 0.0)

    gargs = generator_args(args.cam_h)

    if args.dry_run:
        # Sampling only -- no server, no policy. Checks that the generator rules can actually
        # be satisfied in this scene and prints the resulting distribution, so a bad episode
        # spec is caught in minutes instead of after hours of navigation.
        got = []
        for i in range(args.num_episodes):
            ep = build_episode(sim, pf, esdf, rng, args, gargs)
            if ep is None:
                print(f"  ep{i}: SKIP (no episode satisfied the sampling rules)", flush=True)
                continue
            got.append(ep)
            print(f"  ep{i}: leg1={ep['leg1_m']:.1f}m/{ep['leg1_frames']}fr "
                  f"anchor@{ep['anchor_i']} gap={ep['frame_gap']}fr "
                  f"| revisit geo={ep['geo_back']:.2f}m covis={ep['c_cov']:.2f} "
                  f"head_off={ep['head_off']:.0f}deg "
                  f"| novel geo={ep['geo_novel']:.2f}m covis={ep['n_cov']:.2f}", flush=True)
        print(f"\n=== DRY RUN: {len(got)}/{args.num_episodes} episodes ===", flush=True)
        if got:
            for name, vals in (("revisit geo (m)", [e["geo_back"] for e in got]),
                               ("novel geo (m)", [e["geo_novel"] for e in got]),
                               ("frame gap", [e["frame_gap"] for e in got]),
                               ("revisit covis", [e["c_cov"] for e in got])):
                print(f"  {name:16s} min={min(vals):.2f} mean={np.mean(vals):.2f} "
                      f"max={max(vals):.2f}", flush=True)
            print(f"\n  all gaps >= {EXCLUDE_RECENT}? "
                  f"{all(e['frame_gap'] >= EXCLUDE_RECENT for e in got)}", flush=True)
        return

    rows = []
    for i in range(args.num_episodes):
        # built one at a time: sampling needs leg 1 RENDERED (the covisibility tests read its
        # poses and depths), so episodes cannot be drawn up front from geometry alone.
        ep = build_episode(sim, pf, esdf, rng, args, gargs)
        if ep is None:
            print(f"  ep{i}: SKIP (no episode satisfied the generator sampling rules)", flush=True)
            continue
        print(f"  ep{i}: leg1={ep['leg1_m']:.1f}m/{ep['leg1_frames']}fr  "
              f"anchor@{ep['anchor_i']} (gap={ep['frame_gap']}fr, need>={EXCLUDE_RECENT})  "
              f"revisit: geo={ep['geo_back']:.2f}m covis={ep['c_cov']:.2f} "
              f"head_off={ep['head_off']:.0f}deg | novel: geo={ep['geo_novel']:.2f}m "
              f"covis={ep['n_cov']:.2f}", flush=True)
        row = {"ep": i, "geo_back": ep["geo_back"], "geo_novel": ep["geo_novel"],
               "leg1_m": ep["leg1_m"], "leg1_frames": ep["leg1_frames"],
               "anchor_i": ep["anchor_i"], "frame_gap": ep["frame_gap"],
               "c_cov": ep["c_cov"], "n_cov": ep["n_cov"]}
        for cond in ("revisit", "novel"):
            t0 = time.time()
            print(f"  ep{i} {cond}: starting (replay leg1 then navigate)", flush=True)
            out = run_condition(sim, pf, ep, cond, args.port, args)
            if out is None:
                print(f"  ep{i} {cond}: SKIP", flush=True)
                row[cond] = None
                continue
            r, vframes = out
            if args.save_video and vframes:
                try:
                    import imageio
                    imageio.mimsave(os.path.join(args.out, f"ep{i:02d}_{cond}.mp4"),
                                    vframes, fps=10)
                except Exception as e:
                    print(f"  (video save skipped: {e})", flush=True)
            r["wall_s"] = round(time.time() - t0, 1)
            row[cond] = r
            g = "n/a" if r["gate_mean"] is None else f"{r['gate_mean']:.3f}"
            mh = "n/a" if r["match_hit"] is None else f"{r['match_hit']:.2f}"
            print(f"  ep{i} {cond}: SR={r['success']} d={r['final_dist']:.2f} "
                  f"geo={r['geo']:.2f} gate={g} match_hit={mh} "
                  f"leg1={r['leg1_frames']}fr anchor@{r['c_frame_idx']} gap={r['frame_gap']}fr "
                  f"k={r['k_at_leg3']} in_E(k)={r['c_in_ek']} ({r['wall_s']}s)", flush=True)
        rows.append(row)
        with open(os.path.join(args.out, "rows.json"), "w") as f:
            json.dump(rows, f, indent=2)

    def agg(cond, key):
        v = [r[cond][key] for r in rows if r.get(cond) and r[cond].get(key) is not None]
        return float(np.mean(v)) if v else None

    summary = {
        "n": len(rows),
        "SR_revisit": agg("revisit", "success"),
        "SR_novel": agg("novel", "success"),
        "gate_revisit": agg("revisit", "gate_mean"),
        "gate_novel": agg("novel", "gate_mean"),
        "match_hit_revisit": agg("revisit", "match_hit"),
        "c_in_ek_rate": agg("revisit", "c_in_ek"),
    }
    if summary["SR_revisit"] is not None and summary["SR_novel"] is not None:
        summary["SR_gain"] = summary["SR_revisit"] - summary["SR_novel"]
    if summary["gate_revisit"] is not None and summary["gate_novel"] is not None:
        summary["gate_sep"] = summary["gate_revisit"] - summary["gate_novel"]
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

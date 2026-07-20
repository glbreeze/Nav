#!/usr/bin/env python
"""Closed-loop image-goal navigation eval in Habitat MP3D (same-domain as memnav training).

Talks to the SAME memnav server the Isaac eval uses (navigator_reset + imagegoal_step over
HTTP): the server takes (goal image, current rgb, current depth) and returns a trajectory of
2D waypoints in the robot local ground frame. Here we drive a kinematic Habitat agent along
that trajectory (unicycle motion matching the InternData-N1 fingerprint: constant forward
speed, bounded turn rate, no turn-in-place, navmesh-slide for collisions).

Camera + sim setup are imported from MemNavData/generate_twoleg.py so the eval camera is
byte-for-byte the one that rendered the training/goal images (480x270, hfov ~68).

Axis convention of the returned trajectory (robot frame: is +x forward? +y left?) is set by
env vars so it can be fixed during the smoke test without editing code:
  MEMNAV_FWD_SIGN (+1/-1), MEMNAV_LAT_SIGN (+1/-1), MEMNAV_SWAP_XY (0/1).

Runs in the `habitat` conda env (habitat-sim + requests + numpy + PIL). No torch here --
the model lives in the server (lbm env).
"""
import argparse, os, sys, io, json, time
import numpy as np
from PIL import Image
import requests

# --- import the data-generation sim/camera so eval matches training exactly ---
MEMNAVDATA = os.environ.get("MEMNAVDATA_DIR", "/scratch/ay2710/Nav_memnav/MemNavData")
if MEMNAVDATA not in sys.path:
    sys.path.insert(0, MEMNAVDATA)
import generate_twoleg as G   # noqa: E402  (make_sim, render, agent_state, geodesic, W,H,K,cam consts)
import habitat_sim            # noqa: E402
from mpc_tracking import MPC_Controller   # noqa: E402  (Isaac's exact trajectory-tracking MPC)

DT = 0.1   # MPC control period (matches MPC_Controller.T)

FWD_SIGN = float(os.environ.get("MEMNAV_FWD_SIGN", "1"))
LAT_SIGN = float(os.environ.get("MEMNAV_LAT_SIGN", "1"))
SWAP_XY  = os.environ.get("MEMNAV_SWAP_XY", "0") == "1"
FLIP_IMG = os.environ.get("MEMNAV_FLIP_IMG", "0") == "1"   # horizontally mirror rgb+goal+depth sent to server


# --------------------------------------------------------------------------- #
# HTTP client (same protocol as utils_tasks/client_utils.py, PIL/np encoded)
# --------------------------------------------------------------------------- #
def navigator_reset(intrinsic, stop_threshold, port, env_id=None):
    if env_id is None:
        r = requests.post(f"http://localhost:{port}/navigator_reset",
                          json={'intrinsic': intrinsic.tolist(),
                                'stop_threshold': stop_threshold, 'batch_size': 1})
    else:
        r = requests.post(f"http://localhost:{port}/navigator_reset_env",
                          json={'env_id': env_id})
    return json.loads(r.text)['algo']


def _jpg(rgb):
    b = io.BytesIO(); Image.fromarray(rgb.astype(np.uint8)).save(b, format='JPEG', quality=95)
    return b.getvalue()


def _png_depth(depth_m):
    d = np.clip(depth_m * 10000.0, 0, 65535.0).astype(np.uint16)
    b = io.BytesIO(); Image.fromarray(d).save(b, format='PNG')
    return b.getvalue()


def imagegoal_step(goal_rgb, cur_rgb, cur_depth, port):
    files = {
        'image': ('image.jpg', _jpg(cur_rgb), 'image/jpeg'),
        'goal':  ('goal.jpg',  _jpg(goal_rgb), 'image/jpeg'),
        'depth': ('depth.png', _png_depth(cur_depth), 'image/png'),
    }
    r = requests.post(f"http://localhost:{port}/imagegoal_step",
                      files=files, data={'depth_time': time.time(), 'rgb_time': time.time()})
    j = json.loads(r.text)
    return np.array(j['trajectory']), np.array(j['all_trajectory']), np.array(j['all_values'])


# --------------------------------------------------------------------------- #
# geometry: agent yaw (about +Y up) -> world x-z forward/left unit vectors
# agent_state uses rotation_vector [0,yaw,0]; camera optical -Z fwd, +Y up.
# forward = Ry(yaw)@[0,0,-1] = (-sin, -cos) in (x,z);  left = Ry(yaw)@[-1,0,0] = (-cos, sin)
# --------------------------------------------------------------------------- #
def fwd_xz(yaw):  return np.array([-np.sin(yaw), -np.cos(yaw)])
def left_xz(yaw): return np.array([-np.cos(yaw),  np.sin(yaw)])


def sample_episodes(pf, n, d_min, d_max, rng, floor_tol=0.5, tries=4000):
    """(start_xz, start_y, goal_xz, goal_yaw) with geodesic(start,goal) in [d_min,d_max], same floor."""
    eps = []
    for _ in range(tries):
        if len(eps) >= n:
            break
        a = np.array(pf.get_random_navigable_point())
        b = np.array(pf.get_random_navigable_point())
        if abs(a[1] - b[1]) > floor_tol:
            continue
        ok, gd, _pts = G.geodesic(pf, a, b)
        if not ok or not np.isfinite(gd) or not (d_min <= gd <= d_max):
            continue
        # goal image faces ALONG the arrival direction (last geodesic segment) so it's the
        # natural "what you see when you arrive" view -- a random yaw can face a wall and be
        # unlocalizable, dooming any model. Fall back to start->goal direction if path too short.
        if _pts is not None and len(_pts) >= 2:
            seg = np.asarray(_pts[-1]) - np.asarray(_pts[-2])
        else:
            seg = b - a
        goal_yaw = float(np.arctan2(-seg[0], -seg[2]))
        # start facing roughly toward the goal (+noise) so the goal is visible -> fair
        # test of "follow a visible goal". forward_xz(yaw)=(-sin,-cos) => yaw=atan2(-dx,-dz).
        gdx, gdz = b[0] - a[0], b[2] - a[2]
        face = float(np.arctan2(-gdx, -gdz))
        if os.environ.get("MEMNAV_START_FACE", "1") == "1":
            start_yaw = face + float(rng.uniform(-0.5, 0.5))
        else:
            start_yaw = float(rng.uniform(0, 2 * np.pi))
        eps.append(dict(start=a, goal=b, goal_yaw=goal_yaw, geo=float(gd), start_yaw=start_yaw))
    return eps


def run_episode(sim, pf, ep, port, args):
    cam_h = args.cam_h
    start, goal, goal_yaw = ep['start'], ep['goal'], ep['goal_yaw']
    goal_xz = np.array([goal[0], goal[2]])
    floor_y = float(start[1])

    # render the goal image (canonical view at goal pose)
    goal_rgb, _ = G.render(sim, np.array([goal[0], goal[1] + cam_h, goal[2]]), goal_yaw)

    navigator_reset(G.K, args.stop_threshold, port)  # per-episode fresh memory
    pos = np.array([start[0], start[2]]); yaw = ep['start_yaw']
    path_len = 0.0; stop_run = 0; frames = []
    srv_time = 0.0; srv_n = 0
    reached = False
    for step in range(args.max_steps):
        cam_pos = np.array([pos[0], floor_y + cam_h, pos[1]])
        rgb, depth = G.render(sim, cam_pos, yaw)
        d_goal = float(np.linalg.norm(pos - goal_xz))
        if step % 25 == 0:
            gv = goal_xz - pos
            brg = np.degrees(np.arctan2(float(gv @ left_xz(yaw)), float(gv @ fwd_xz(yaw))))
            avg_ms = 1000.0 * srv_time / max(srv_n, 1)
            print(f"    step {step:3d} d_goal={d_goal:.2f} goal_bearing={brg:.0f}deg path={path_len:.1f} "
                  f"srv={avg_ms:.0f}ms/step", flush=True)
        if d_goal < args.success_dist:
            reached = True; break
        if args.save_video and step % args.vid_stride == 0:
            frames.append(_overlay(rgb, goal_rgb, d_goal, step))

        _t0 = time.time()
        if FLIP_IMG:
            traj, _, _ = imagegoal_step(goal_rgb[:, ::-1], rgb[:, ::-1], depth[:, ::-1], port)
        else:
            traj, _, _ = imagegoal_step(goal_rgb, rgb, depth, port)
        srv_time += time.time() - _t0; srv_n += 1
        pts3 = np.array(traj).reshape(-1, 3)          # server returns (1,24,3) = xyz waypoints
        # WARMUP: server returns the (0,0.1,0) placeholder until its streaming memory has
        # >= num_scale+window-1 frames. During warmup just go STRAIGHT to fill the stream
        # (spinning here would poison the memory with in-place views).
        is_warmup = np.allclose(pts3, np.array([0.0, 0.1, 0.0]), atol=1e-3)
        if 15 <= step <= 22:
            # DECISIVE: does the MODEL point at the goal right after warmup (robot still ~facing it)?
            gv = goal_xz - pos
            tf = float(gv @ fwd_xz(yaw)); tl = float(gv @ left_xz(yaw))
            true_brg = np.degrees(np.arctan2(tl, tf))
            mfx = float(pts3[-1, 0]); mlat = float(pts3[-1, 1])          # raw model (index0=fwd,index1=lat)
            model_brg = np.degrees(np.arctan2(mlat, mfx))
            print(f"    [diag s{step}] MODEL fwd={mfx:.2f} lat={mlat:.2f} dir={model_brg:.0f}deg  "
                  f"vs  TRUE goal dir={true_brg:.0f}deg   (agree? {abs(model_brg-true_brg)<45})", flush=True)

        if is_warmup:
            # stream still filling (~15 steps). Start faces the goal -> drive STRAIGHT to fill it.
            v_cmd, w_cmd = args.v_max, 0.0
        else:
            if pts3.size == 0 or np.abs(pts3).max() < 1e-3:
                stop_run += 1
                if stop_run >= args.stop_patience:
                    break
                continue
            stop_run = 0
            fx = FWD_SIGN * pts3[:, 0]; lat = LAT_SIGN * pts3[:, 1]   # index0=fwd, index1=lat
            if SWAP_XY:
                fx, lat = lat.copy(), fx.copy()
            # place the model trajectory in WORLD (Habitat x,z) then MPC-track it, exactly like Isaac
            fwd = fwd_xz(yaw); lft = left_xz(yaw)
            W = pos[None, :] + fx[:, None] * fwd[None, :] + lat[:, None] * lft[None, :]   # (24,2)
            W = np.vstack([pos[None, :], W])                          # path starts at current pos
            theta = float(np.arctan2(-np.cos(yaw), -np.sin(yaw)))     # MPC heading: (cos,sin)=fwd_xz
            try:
                mpc = MPC_Controller(W, N=15, desired_v=args.v_max, v_max=args.v_max,
                                     w_max=args.w_max, ref_gap=3)
                u, _ = mpc.solve(np.array([pos[0], pos[1], theta]))
                v_cmd, w_cmd = float(u[0, 0]), float(u[0, 1])
            except Exception as e:
                if step % 25 == 0:
                    print(f"    (mpc solve fail @s{step}: {e})", flush=True)
                v_cmd, w_cmd = 0.0, 0.0
        # integrate unicycle: advance along fwd_xz(yaw); yaw -= w*dt (reflection MPC theta<->Habitat yaw)
        proposed = pos + v_cmd * DT * fwd_xz(yaw)
        yaw = yaw - w_cmd * DT
        # navmesh slide (collision): try_step in habitat 3D
        a3 = np.array([pos[0], floor_y, pos[1]])
        b3 = np.array([proposed[0], floor_y, proposed[1]])
        try:
            nxt = np.array(pf.try_step(a3, b3))          # slide along navmesh (no wall crossing)
        except Exception:
            snap = np.array(pf.snap_point(b3))            # fallback: snap, reject big jumps
            nxt = snap if np.linalg.norm(snap[[0, 2]] - b3[[0, 2]]) < args.v_max * DT * 2 else a3
        new_pos = np.array([nxt[0], nxt[2]])
        path_len += float(np.linalg.norm(new_pos - pos))
        pos = new_pos

    d_final = float(np.linalg.norm(pos - goal_xz))
    reached = reached or (d_final < args.success_dist)
    geo = ep['geo']
    spl = (geo / max(path_len, geo)) if reached and path_len > 0 else 0.0
    return dict(success=int(reached), spl=float(spl), geo=geo,
                path_len=path_len, final_dist=d_final, steps=step + 1), frames


def _overlay(rgb, goal_rgb, d, step):
    h = min(rgb.shape[0], goal_rgb.shape[0])
    return np.concatenate([rgb[:h], goal_rgb[:h]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--navmesh", default="")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--num_episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--success_dist", type=float, default=1.5)   # parity with Isaac eval
    ap.add_argument("--speed", type=float, default=0.0376)       # legacy (unused by MPC path)
    ap.add_argument("--v_max", type=float, default=0.4)          # MPC linear speed (m/s)
    ap.add_argument("--w_max", type=float, default=1.2)          # MPC angular speed (rad/s)
    ap.add_argument("--max_turn", type=float, default=np.deg2rad(4.5))
    ap.add_argument("--warmup_turn", type=float, default=np.deg2rad(9.0),
                    help="in-place rotation per step while the stream fills (scan, no translate)")
    ap.add_argument("--lookahead", type=float, default=0.6)
    ap.add_argument("--cam_h", type=float, default=0.5)
    ap.add_argument("--stop_threshold", type=float, default=-0.5)
    ap.add_argument("--stop_patience", type=int, default=8)
    ap.add_argument("--d_min", type=float, default=3.0)
    ap.add_argument("--d_max", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="./imagegoal_habitat_out")
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--vid_stride", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    nav = args.navmesh or args.scene.replace(".glb", ".navmesh")
    sim = G.make_sim(args.scene, nav, agent_radius=0.30, agent_height=1.5)
    pf = sim.pathfinder
    assert pf.is_loaded, "navmesh not loaded"

    algo = navigator_reset(G.K, args.stop_threshold, args.port)
    print(f"[eval] server algo={algo} axis(fwd={FWD_SIGN},lat={LAT_SIGN},swap={SWAP_XY})")

    eps = sample_episodes(pf, args.num_episodes, args.d_min, args.d_max, rng)
    print(f"[eval] sampled {len(eps)} episodes on {os.path.basename(args.scene)}")

    metrics = []
    for i, ep in enumerate(eps):
        m, frames = run_episode(sim, pf, ep, args.port, args)
        metrics.append(m)
        print(f"[ep {i:02d}] success={m['success']} spl={m['spl']:.3f} "
              f"geo={m['geo']:.1f} path={m['path_len']:.1f} final_d={m['final_dist']:.2f} steps={m['steps']}")
        if args.save_video and frames:
            try:
                import imageio
                imageio.mimsave(os.path.join(args.out, f"ep_{i:02d}.mp4"), frames, fps=10)
            except Exception as e:
                print(f"  (video save skipped: {e})")

    sr = np.mean([m['success'] for m in metrics]) if metrics else 0.0
    spl = np.mean([m['spl'] for m in metrics]) if metrics else 0.0
    summary = dict(scene=os.path.basename(args.scene), n=len(metrics),
                   SR=float(sr), SPL=float(spl))
    json.dump({'summary': summary, 'per_episode': metrics},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(f"\n===== RESULT {summary['scene']}: SR={sr:.3f} SPL={spl:.3f} over {len(metrics)} eps =====")
    sim.close()


if __name__ == "__main__":
    main()

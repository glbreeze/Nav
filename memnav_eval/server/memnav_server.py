from PIL import Image
from flask import Flask, jsonify, request
import argparse
import atexit
import cv2
import datetime
import imageio
import numpy as np
import os
import traceback

from policy_agent import MemNav_Agent

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8888)
parser.add_argument("--checkpoint", type=str, default="./memnav.ckpt")
parser.add_argument("--temporal_depth", type=int, default=8)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--vid_dir", type=str, default=os.environ.get(
    "MEMNAV_VID_DIR", "/home/yl13095/new_Nav/log"))
parser.add_argument("--lingbot_repo", type=str, default=os.environ.get(
    "LINGBOT_REPO", "/scratch/yl13095/Nav/lingbot-map-repo"))
parser.add_argument("--lingbot_weights", type=str, default=os.environ.get(
    "LINGBOT_WEIGHTS", "/scratch/yl13095/Nav/checkpoints/lingbot-map/lingbot-map-long.pt"))
args = parser.parse_known_args()[0]

app = Flask(__name__)
navigator = None
fps_writer = None
fps_path = None
step_idx = 0


def _close_fps_writer():
    global fps_writer
    if fps_writer is not None:
        try:
            fps_writer.close()
        except Exception:
            pass
        fps_writer = None


atexit.register(_close_fps_writer)


def _safe_append_frame(writer, frame):
    if writer is None or frame is None:
        return
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 2, 3, 4):
        return
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    writer.append_data(arr)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "algo": "memnav"})


@app.route("/navigator_reset", methods=["POST"])
def navigator_reset():
    global navigator, fps_writer, fps_path, step_idx
    intrinsic = np.array(request.get_json().get("intrinsic"))
    threshold = np.array(request.get_json().get("stop_threshold"))
    batchsize = int(request.get_json().get("batch_size"))
    if navigator is None:
        navigator = MemNav_Agent(
            intrinsic,
            # MUST match how the checkpoint's caches were precomputed (mp3d w32 run: 32).
            # This is the LingBot sliding window: get it wrong and the scale/history/window
            # regions are partitioned differently than at train time. Same env var the
            # training config reads, so one export keeps both sides in sync.
            memory_size=int(os.environ.get("MEMNAV_WINDOW", "32")),
            predict_size=24,
            temporal_depth=args.temporal_depth,
            heads=8,
            token_dim=384,
            navi_model=args.checkpoint,
            device=args.device,
            lingbot_repo=args.lingbot_repo,
            lingbot_weights=args.lingbot_weights,
        )
    navigator.reset(batchsize, threshold)
    if fps_writer is not None:
        fps_writer.close()
    os.makedirs(args.vid_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fps_path = os.path.join(args.vid_dir, f"memnav_traj_{ts}.mp4")
    fps_writer = imageio.get_writer(fps_path, fps=7)
    step_idx = 0
    print(f"[memnav_server] trajectory video -> {fps_path}", flush=True)
    return jsonify({"algo": "memnav"})


@app.route("/navigator_reset_env", methods=["POST"])
def navigator_reset_env():
    navigator.reset_env(int(request.get_json().get("env_id")))
    return jsonify({"algo": "memnav"})


@app.route("/stream_frame", methods=["POST"])
def stream_frame():
    """Append a rendered frame to memory without running the policy.

    The revisit eval drives leg 1 (walk past the future goal) and leg 2 (walk away) along a
    GT path, so the goal is genuinely in memory -- and >= exclude_recent frames back, i.e.
    inside E(k) -- before the policy is asked to navigate to it. imagegoal_step cannot be
    used for this: it needs a goal image, and the goal CLS is cached on first call.
    """
    global navigator
    try:
        image = np.asarray(Image.open(request.files["image"].stream).convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image.reshape((navigator.batch_size, -1, image.shape[1], 3))
        k = navigator.stream_frame(image)
        return jsonify({"k": int(k)})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/set_goal", methods=["POST"])
def set_goal():
    """Install the goal image AFTER the memory is built (revisit eval leg 3). Unlike
    navigator_reset_env this keeps the stream state."""
    global navigator
    try:
        goal = np.asarray(Image.open(request.files["goal"].stream).convert("RGB"))
        goal = cv2.cvtColor(goal, cv2.COLOR_RGB2BGR)
        goal = goal.reshape((navigator.batch_size, -1, goal.shape[1], 3))
        navigator.set_goal(goal)
        return jsonify({"ok": True})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/imagegoal_step", methods=["POST"])
def imagegoal_step():
    global navigator, fps_writer, step_idx
    try:
        image = np.asarray(Image.open(request.files["image"].stream).convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image.reshape((navigator.batch_size, -1, image.shape[1], 3))

        goal = np.asarray(Image.open(request.files["goal"].stream).convert("RGB"))
        goal = cv2.cvtColor(goal, cv2.COLOR_RGB2BGR)
        goal = goal.reshape((navigator.batch_size, -1, goal.shape[1], 3))

        depth = np.asarray(Image.open(request.files["depth"].stream).convert("I"))
        depth = (depth.astype(np.float32) / 10000.0)[:, :, np.newaxis]
        depth = depth.reshape((navigator.batch_size, -1, depth.shape[1], 1))

        execute, all_traj, all_vals, mask = navigator.step_imagegoal(goal, image, depth)
        _safe_append_frame(fps_writer, mask)
        step_idx += 1
        return jsonify({
            "trajectory": execute.tolist(),
            "all_trajectory": all_traj.tolist(),
            "all_values": all_vals.tolist(),
            # retrieval readout: P(revisit) and which memory frame was matched. The revisit
            # eval scores these directly against ground truth (it knows the goal's frame
            # index), which SR cannot do -- SR is downstream of MPC and trajectory choice.
            "revisit_gate": navigator._last_gate,
            "match_idx": navigator._last_match,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, threaded=False)

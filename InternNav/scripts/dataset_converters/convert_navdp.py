#!/usr/bin/env python
"""
Convert one InternData-N1 scene tarball into the layout NavDP's training loader
(`internnav/dataset/navdp_dataset_lerobot.py`) expects.

Source (HF tar.gz content):
    <scene>/
      ├── meta/pointcloud.ply                  # scene-level pcd, gray surface + dim blue obstacles
      ├── data/chunk-000/episode_NNN.parquet   # one parquet per episode (columns:
      │                                          observation.camera_{intrinsic,extrinsic}, action)
      └── videos/chunk-000/
          ├── observation.images.rgb/episode_NNN_MMM.jpg
          └── observation.images.depth/episode_NNN_MMM.png

Target (one traj_dir per episode, matching loader walk root/group/scene/traj):
    <out_root>/<group>/<scene>/ep_NNN/
      ├── data/chunk-000/episode_000000.parquet      (symlink to source episode_NNN.parquet)
      ├── data/chunk-000/path.ply                    (generated: black=episode path, blue=scene obstacles)
      └── videos/chunk-000/observation.images.{rgb,depth}/{0..K-1}.{jpg,png}   (symlinks, renumbered)

NOTE: the loader also has a latent bug — `process_data_parquet` builds
    camera_trajectory = np.array([np.stack(frame) for frame in df['action']])
which with current HF data yields shape (N, 16), not (N, 4, 4). Apply this
one-line patch in navdp_dataset_lerobot.py line 177:
    camera_trajectory = np.array([np.stack(frame).reshape(4, 4) for frame in df['action']], dtype=np.float64)
"""
import argparse
import re
import tarfile
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

RGB_RE = re.compile(r'episode_(\d+)_(\d+)\.jpg$')
DEP_RE = re.compile(r'episode_(\d+)_(\d+)\.png$')
PQ_RE = re.compile(r'episode_(\d+)\.parquet$')


def extract_tarball(tarball: Path, extract_root: Path) -> Path:
    scene = tarball.name.replace('.tar.gz', '').replace('.tgz', '')
    target = extract_root / scene
    extract_root.mkdir(parents=True, exist_ok=True)
    if target.exists() and any(target.iterdir()):
        print(f'[extract] SKIP (already present): {target}')
        return target
    print(f'[extract] {tarball}  ->  {extract_root}')
    with tarfile.open(tarball, 'r:gz') as tf:
        tf.extractall(extract_root)
    return target


def build_path_ply(parquet_path: Path, scene_pcd_path: Path, out_ply: Path):
    df = pd.read_parquet(parquet_path)
    acts = np.array([np.stack(f) for f in df['action']], dtype=np.float64).reshape(-1, 4, 4)
    path_xyz = acts[:, 0:3, 3]

    scene = o3d.io.read_point_cloud(str(scene_pcd_path))
    scene_pts = np.asarray(scene.points)
    scene_cols = np.asarray(scene.colors)
    blue_mask = np.abs(scene_cols - np.array([0, 0, 0.5])).sum(axis=-1) < 0.05
    obs_xyz = scene_pts[blue_mask]

    all_xyz = np.vstack([path_xyz, obs_xyz])
    all_col = np.vstack([
        np.zeros_like(path_xyz),
        np.tile(np.array([0, 0, 0.5]), (len(obs_xyz), 1)),
    ])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_xyz)
    pcd.colors = o3d.utility.Vector3dVector(all_col)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_ply), pcd)


def link(dst: Path, src: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def convert_scene(scene_root: Path, group: str, out_root: Path):
    scene = scene_root.name
    data_dir = scene_root / 'data' / 'chunk-000'
    rgb_dir = scene_root / 'videos' / 'chunk-000' / 'observation.images.rgb'
    depth_dir = scene_root / 'videos' / 'chunk-000' / 'observation.images.depth'
    pcd_path = scene_root / 'meta' / 'pointcloud.ply'

    pq_files = sorted(data_dir.glob('episode_*.parquet'))
    if not pq_files:
        raise SystemExit(f'No parquet files in {data_dir}')
    if not pcd_path.exists():
        raise SystemExit(f'Missing {pcd_path}')

    rgb_by_ep, dep_by_ep = {}, {}
    for p in rgb_dir.iterdir():
        m = RGB_RE.match(p.name)
        if m:
            rgb_by_ep.setdefault(int(m.group(1)), []).append((int(m.group(2)), p))
    for p in depth_dir.iterdir():
        m = DEP_RE.match(p.name)
        if m:
            dep_by_ep.setdefault(int(m.group(1)), []).append((int(m.group(2)), p))

    for pq in pq_files:
        ep = int(PQ_RE.match(pq.name).group(1))
        traj_dir = out_root / group / scene / f'ep_{ep:03d}'

        link(traj_dir / 'data' / 'chunk-000' / 'episode_000000.parquet', pq)
        build_path_ply(pq, pcd_path, traj_dir / 'data' / 'chunk-000' / 'path.ply')

        rgb_sorted = sorted(rgb_by_ep.get(ep, []))
        dep_sorted = sorted(dep_by_ep.get(ep, []))
        if len(rgb_sorted) != len(dep_sorted):
            print(f'  WARN ep {ep}: rgb={len(rgb_sorted)} depth={len(dep_sorted)} — skipping')
            continue
        for i, (_, src) in enumerate(rgb_sorted):
            link(traj_dir / 'videos' / 'chunk-000' / 'observation.images.rgb' / f'{i}.jpg', src)
        for i, (_, src) in enumerate(dep_sorted):
            link(traj_dir / 'videos' / 'chunk-000' / 'observation.images.depth' / f'{i}.png', src)
        print(f'[ok] {group}/{scene}/ep_{ep:03d}  ({len(rgb_sorted)} frames)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tarball', type=Path, required=True, help='path to *.tar.gz')
    ap.add_argument('--group', required=True, help='e.g. replica_zed, 3dfront_d435i')
    ap.add_argument('--extract-root', type=Path, required=True,
                    help='dir where the tarball is expanded to <scene>/')
    ap.add_argument('--out-root', type=Path, required=True,
                    help='root_dir to pass to the NavDP loader')
    args = ap.parse_args()

    scene_root = extract_tarball(args.tarball, args.extract_root)
    convert_scene(scene_root, args.group, args.out_root)
    print(f'\nDone. Set config.il.root_dir = {args.out_root}')


if __name__ == '__main__':
    main()
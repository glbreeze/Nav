# memnav_eval — closed-loop Habitat eval for MemNav

Self-contained. Everything needed to run the revisit-vs-novel experiment and to reproduce
every number in `FINDINGS.md` is in this directory.

```
eval_revisit_habitat.py     revisit vs novel, matched controls   <- the memory experiment
eval_imagegoal_habitat.py   plain image-goal (novel branch only)
mpc_tracking.py             trajectory-tracking MPC (client-side)
server/                     Flask server + agent + streaming state
sbatch/                     submission scripts (paths are NYU-Torch specific)
diagnostics/                one script per measured claim in FINDINGS.md
```

The model-side changes these depend on are NOT here — they are modifications to
`InternNav/internnav/model/basemodel/memnav/{memnav_policy,lingbot_stream}.py` in this same
commit, since they have to stay in place to be usable.

---

## What the two evals measure

`eval_imagegoal_habitat.py` samples a random (start, goal) pair, so the goal is somewhere the
robot has never been: the gate says novel, revisit tokens get masked, memory is never
consulted. It measures the part MemNav shares with any image-goal policy.

`eval_revisit_habitat.py` mirrors how the training data is built, so the eval distribution
matches what the checkpoint was fit on:

```
leg 1   start ---> A     GT path, streamed into memory. The robot walks PAST the goal.
leg 2   A     ---> B     the policy drives, given B's image.      <- the measured part
```

`B` comes from `generate_twoleg.sample_revisit`, which is CALLED, not reimplemented — pick a
frame X the robot occupied on leg 1, jitter the goal inside a 1.5 m disk around X, take X's
heading ±45°, require co-visibility in [0.20, 1.00]. The novel control comes from
`sample_novel` and is **distance-matched** to the revisit goal (±0.5 m), so `SR_gain` cannot
be explained by "near vs far" — without that match the novel goals averaged 6.77 m against
revisit's 2.47 m.

Both conditions replay identical leg-1 frames from the same start pose. The only variable is
whether the goal is in memory.

Success is position-only: `|pos - goal_xz| < success_dist`. No heading term.

## Metrics

`metrics.json`:

| key | meaning |
|---|---|
| `SR_revisit` / `SR_novel` / `SR_gain` | the headline comparison |
| `gate_revisit` / `gate_novel` / `gate_sep` | does the gate open when it should |
| `match_hit_revisit` | is the retrieved frame actually near the anchor (ground truth) |
| `c_in_ek_rate` | self-check: did the target really land inside E(k)? if not ≈1, the run is void |

SR sits downstream of MPC, collision sliding and trajectory choice, so it can hide working
retrieval or flatter broken retrieval. `gate` and `match_hit` are the direct measurements.

## Running

```bash
# 1. sanity, no habitat, ~3 min
MEMNAV_CKPT=<...>/memnav.ckpt sbatch sbatch/online_smoke.sbatch

# 2. sampling only, no policy -- checks the episode spec is satisfiable, ~2 min
sbatch sbatch/dryrun.sbatch

# 3. the experiment
sbatch --export=ALL,MEMNAV_CKPT=<...>/memnav.ckpt,MEMNAV_SWAP_XY=1,NUM_EPISODES=6,\
MAX_STEPS=120,SUCCESS_DIST=1.0,MEMNAV_DBACK_HI=3.0 sbatch/eval_revisit.sbatch
```

Required env (set inside the sbatch files; change the paths for another cluster):

```
MEMNAV_SWAP_XY=1        REQUIRED. The habitat client's yaw->axis mapping is transposed;
                        with 0 the robot spins in place and every episode scores 0.
MEMNAV_WINDOW=32        must match how the caches were precomputed
MEMNAV_NUM_SCALE=8
MEMNAV_MAX_FRAME_NUM=2048
MEMNAV_EXCLUDE_RECENT=83
LINGBOT_REPO / LINGBOT_WEIGHTS      frozen backbone (not in the checkpoint)
```

Checkpoints hold only the trainable heads (`memnav.ckpt`); LingBot is loaded separately from
`LINGBOT_WEIGHTS` and `load_state_dict(..., strict=False)` is expected to report ~2611 missing
keys. That is by design, not an error.

## Tunables worth knowing

```
MEMNAV_DBACK_HI=3.0     cap on the leg-2 return distance. The policy has never solved
                        anything past ~2.3 m, so a larger cap makes both conditions fail and
                        SR_gain uninformative.
MEMNAV_MIN_TRAJ_M=0     zeroes translation on short predicted trajectories. Set to 0.
                        At the old default of 0.5 it fired on ~2/3 of steps and, because the
                        driver reads only channels 0 and 1 (the yaw it preserves is never
                        read), collapsed every waypoint onto the current position -- the robot
                        froze, and `abs(pts3).max()` stayed non-zero so the stop-patience
                        early exit never triggered either.
MEMNAV_MATCH_TOL_M=0.5  distance-matching tolerance for the novel control
MEMNAV_SAMPLE_NUM=8     diffusion samples. Measured: 1 and 8 cost the same (7504 vs 7561 ms)
                        -- they are batched through one loop -- so lowering it saves nothing.
```

## Known-good and known-broken

Working, with evidence:

- retrieval — `match_hit` 1.00 on most episodes
- the gate — 0.68 on revisit vs 0.09 on novel
- episode construction — `c_in_ek_rate` 1.00, frame gaps 88–156 against a required 83
- pose plumbing — online-built `cam_pose_enc` matches the precomputed disk values to 0.0078 m

Open, and the reason this is being shared:

**The policy predicts trajectory endpoints ~0.1 m long against goals 2–3 m away.** The robot
therefore barely moves — 0.2–0.6 m over 100 steps — even when the predicted heading is right
(5/8 within 45° on revisit; the best was 3° off). Present at every checkpoint tried and in
every run recorded in `FINDINGS.md`, before and after the fixes in this commit.

Retrieval and the gate look healthy, so the gap sits between "knows where the goal is" and
"emits an action that goes there".

⚠️ **Do not read the earlier `SR=0.625` from `eval_imagegoal_habitat` as policy performance.**
All five successes landed at steps 14–37, below `anchor_margin=39` — i.e. while the client was
still driving straight through warm-up. All 56 diagnostic lines in that run are the warm-up
placeholder `fwd=0.00 lat=0.10`. The three episodes where the policy did engage ran to the
200-step cap. That number measures how often the goal happened to be straight ahead.

## Diagnostics

Each script isolates one claim in `FINDINGS.md` and prints a verdict.

| script | question | measured answer |
|---|---|---|
| `online_smoke.py` | does the online inference path run at all | pose vs disk 0.0078 m, SMOKE OK |
| `check_preproc.py` | do memory frames and goal images share a preprocessing path | they did not; retrieval 8/24 vs 24/24 matched |
| `check_goalpose.py` | is the goal's pose correct | error grows with injected compressed history: 0° at n_hist=0, 172° at 100 |
| `check_cache_pollution.py` | do policy calls corrupt the live KV cache | they did; 0.0078 m clean vs 0.6250 m interleaved |
| `sweep_warm.py` | does raising `goal_warm` fix the pose error, and at what cost | yes at n_hist=0; 9.3 s -> 26.7 s per call |

## Caveats in the harness itself

- The step-15..22 heading diagnostic reports `agree` for zero-length outputs, because
  `arctan2(0, 0)` returns −180° which can land within the 45° threshold by accident. Ignore
  `agree` on any line where `fwd` and `lat` are both ~0.
- The 45° threshold is arbitrary.
- `eval_revisit_habitat.py` writes video only with `--save_video`; panels are
  `[what it sees | what it was asked to find | what it retrieved]`.

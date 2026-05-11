
## Scene Categories vs. Path Length for Point-Goal Navigation

| Scene | Path lengths | Format | Notes |
|---|---|---|---|
| `cluttered_easy/hard` | Short (single room) | Single tarball | Quick smoke tests |
| `internscenes_home` | Medium (multi-room) | HF tree, many files | Realistic homes |
| `internscenes_commercial` | Long (large buildings) | Single tarball | Best for long-horizon |

Evaluation Metrics
| Metric | Full name | Definition (this codebase) | Range | Higher/Lower better |
|---|---|---|---|---|
| `success` | Success | 1 if final distance to goal < 1.5 m, else 0 | `{0, 1}` | higher |
| `SPL` | Success-weighted Path Length | `success × min(optimal_dist / actual_dist, 1)` | `[0, 1]` | higher |
| `NE` | Navigation Error | Euclidean distance from final robot position to goal (meters) | `≥ 0` | lower |
| `LE` | Last/Localization Error | Robot's belief vs. truth at termination — diverges when implicit localization drifts | `≥ 0` | lower |

## Result on cluttered_hard

The result is based on evaluation on hard_7 (one of 10 cluttered_hard scenes) with 20 episodes.  

### Results on official ckpt
```bash
/home/asus/Research/Nav/NavDP/startgoal_logoplanner_cluttered_hard_OFFICIAL/hard_7
```

| Metric | Value |
|---|---|
| Success rate | 23.8% (5/21) |
| Mean SPL | 0.227 |
| Mean NE | 7.82 m |




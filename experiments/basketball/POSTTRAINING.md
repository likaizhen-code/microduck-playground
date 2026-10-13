# Post-training lineage (2026-10)

Post-training continuation of the released blind basketball policy (b11,
iteration 6999) on top of the upstream recipe. All numbers use the release
matched protocol: 3 seeds × 1024 environments × 60 s, first fall = failure,
±0.09 m/s planar velocity pushes every 0.5–1 s, commands ±0.30/±0.20/1.0
(evaluation stress scale). Evaluation records: `eval/posttrain_*.json`.

| Stage | Change | Survival | Yaw MAE (rad/s) | Note |
|---|---|---:|---:|---|
| b11 (release) | — | 97.01% | 1.262 | starting point |
| stage 0 `dense_7498` | push interval 1.5–3 → 0.5–1.5 s | 98.99% | 1.226 | failures were push-driven; aligning practice to exam strength |
| stage 0.5 `cmd15_7997` | command scale ×1.5 | 98.99% | 1.193 | yaw hypothesis mostly eliminated (×2.0: 98.60%, yaw unchanged) |
| belief v1 `7997` | ball-state head, weight 0.5, no anchor | 22.2% | 1.298 | aux gradient wrecked the policy (entropy 2.74→3.04) |
| belief v2 `7997` | weight 0.05 + 100-iter head-only warmup | 85.6% | 1.177 | better, still a bad trade |
| **belief v3 `8375`** | **+ KL anchor to frozen source (µ=1.0)** | **99.48%** | **1.134** | Pareto over every metric; exceeds the sighted b2 reference (99.12%) |
| belief v3 `8496` | same, full 500 iters | 99.32% | 1.150 | mid checkpoint 8375 selected |

Warmup-phase measurement (head-only, detached latent): prediction MSE
plateaus ≈ 2.1 = predicting the mean — the baseline LSTM latent contains
almost no ball-state information. That measurement motivated stage 1.

## The anchor lesson

Changing *what is practiced* (data distribution) does not fight the objective
and needs no anchor. Changing *what is optimized* (adding an auxiliary loss)
competes with PPO for the same LSTM parameters and, unanchored, walks the
policy off its optimum at any useful weight (0.5 → 22%, 0.05 → 86%). Freezing
a copy of the source policy and penalizing KL(reference ‖ current) at every
minibatch makes the same pressure safe: survival held above baseline while
yaw and action smoothness improved together.

## Implementation

- `src/mjlab_microduck/basketball_belief.py` — `BasketballBeliefPPO`:
  PPO + auxiliary ball-state head (labels = raw critic obs `55:61`, i.e.
  `basketball_state`) + KL anchor to a frozen deep copy taken at checkpoint
  load. The head lives algorithm-side with its own optimizer so released
  `actor_state_dict`/Adam states load strictly and ONNX export is unchanged.
- `scripts/finetune_basketball_belief.py` — continuation entry point
  (`--belief-weight/--belief-warmup/--anchor-weight`).

## Reproduction

```bash
# stage 0: push-dense continuation from artifacts/basketball/checkpoint.pt (b11)
uv run python scripts/finetune_basketball.py artifacts/basketball/checkpoint.pt \
  --run-name stage0 --num-envs 4096 --iterations 500 --learning-rate 2e-5 \
  --action-rate-weight -0.2 --command-scale 1 --push-interval-s 0.5 1.5 --save-interval 125

# stage 1: belief + anchor from the stage-0 selection
uv run python scripts/finetune_basketball_belief.py logs/rsl_rl/basketball/<stage0>/model_7498.pt \
  --run-name stage1 --num-envs 4096 --iterations 500 --learning-rate 2e-5 \
  --action-rate-weight -0.2 --command-scale 1.5 --push-interval-s 0.5 1.5 \
  --save-interval 125 --belief-weight 0.05 --belief-warmup 100 --anchor-weight 1.0

# matched evaluation (per seed 101/202/303)
uv run python scripts/eval_basketball_long.py <ckpt> out.json --seed 101 \
  --seconds 60 --num-envs 1024 --command-scale 2 --blind
```

Training source paths and per-run manifests live under
`logs/rsl_rl/basketball/` (gitignored); selected artifacts: stage 0 =
`2026-10-05_04-33-16_basketball-push-dense/model_7498.pt`, stage 1 =
`2026-10-07_12-46-44_basketball-belief-v3/model_8375.pt`.

All results are simulation-only; the hardware path is unchanged from the
release (LSTM runtime support required, onboard timing unmeasured).

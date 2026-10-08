"""Belief-head continuation: resume the blind basketball policy and train the
actor's recurrent latent to predict the privileged ball state
(``mjlab_microduck.basketball_belief.BasketballBeliefPPO``).

Recommended source: the phase-0 push-dense checkpoint (best survival), so the
belief stage only has to move the tracking metrics.

  uv run python scripts/finetune_basketball_belief.py \
      logs/rsl_rl/basketball/<push-dense-run>/model_7498.pt \
      --run-name basketball-belief-smoke --num-envs 64 --iterations 5 \
      --push-interval-s 0.5 1.5
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.basketball_distillation import checkpoint_sha256
from mjlab_microduck.basketball_finetune import BasketballContinuationRunner, continuation_configs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--run-name', required=True)
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--action-rate-weight', type=float, default=-.2)
    p.add_argument('--command-scale', type=float, default=1.)
    p.add_argument('--episode-seconds', type=float, default=10.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--push-interval-s', type=float, nargs=2)
    p.add_argument('--save-interval', type=int, default=125)
    p.add_argument('--belief-weight', type=float, default=0.05,
                   help='auxiliary ball-state MSE weight added to the PPO loss '
                        '(0.5 destroyed the balance policy; keep small)')
    p.add_argument('--belief-hidden', type=int, default=64)
    p.add_argument('--belief-lr', type=float, default=1e-3)
    p.add_argument('--belief-warmup', type=int, default=100,
                   help='head-only iterations before the aux gradient reaches the actor')
    p.add_argument('--anchor-weight', type=float, default=1.0,
                   help='KL(current, frozen source policy) penalty weight; 0 disables')
    a = p.parse_args()
    if min(a.num_envs, a.iterations, a.save_interval) < 1:
        p.error('num-envs, iterations and save-interval must be positive')
    if not 0 <= a.belief_weight:
        p.error('belief-weight must be non-negative')
    configure_torch_backends()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg, agent = continuation_configs(checkpoint, learning_rate=a.learning_rate,
        action_rate_weight=a.action_rate_weight, command_scale=a.command_scale,
        episode_seconds=a.episode_seconds, seed=a.seed, push_interval_s=a.push_interval_s)
    agent['algorithm']['class_name'] = 'mjlab_microduck.basketball_belief.BasketballBeliefPPO'
    agent['algorithm']['belief_weight'] = a.belief_weight
    agent['algorithm']['belief_hidden'] = a.belief_hidden
    agent['algorithm']['belief_lr'] = a.belief_lr
    agent['algorithm']['belief_warmup'] = a.belief_warmup
    agent['algorithm']['anchor_weight'] = a.anchor_weight
    cfg.scene.num_envs = a.num_envs
    agent.update(run_name=a.run_name, max_iterations=a.iterations, save_interval=a.save_interval)
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = Path('logs/rsl_rl/basketball') / f'{stamp}_{a.run_name}'
    dump_yaml(log_dir/'params/env.yaml', asdict(cfg))
    dump_yaml(log_dir/'params/agent.yaml', deepcopy(agent))
    manifest = {**vars(a), 'checkpoint':str(a.checkpoint.resolve()),
        'source_sha256':checkpoint_sha256(a.checkpoint), 'source_iteration':checkpoint['iter'],
        'source_optimizer_lr':[g['lr'] for g in checkpoint['optimizer_state_dict']['param_groups']],
        'ball_state_actor':False, 'actor_width':61, 'ball_hold':0., 'schedule':'fixed',
        'algorithm':'BasketballBeliefPPO',
        'belief_label':'critic obs [55:61] (basketball_state, raw)'}
    (log_dir/'manifest.json').write_text(json.dumps(manifest, indent=2))
    raw = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    env = RslRlVecEnvWrapper(raw, clip_actions=agent['clip_actions'])
    try:
        runner = BasketballContinuationRunner(env, agent, str(log_dir), 'cuda:0')
        runner.load(str(a.checkpoint), map_location='cuda:0')
        assert runner.alg.schedule == 'fixed'
        assert all(g['lr'] == a.learning_rate for g in runner.alg.optimizer.param_groups)
        assert raw.common_step_counter >= checkpoint['iter'] * agent['num_steps_per_env']
        env.reset()
        obs = env.get_observations()['actor']
        assert obs.shape == (a.num_envs,61) and torch.count_nonzero(obs[:,55:]) == 0
        runner.add_git_repo_to_log(__file__)
        print('BELIEF_FINETUNE_START', json.dumps({'log_dir':str(log_dir), **manifest}), flush=True)
        runner.learn(a.iterations, init_at_random_ep_len=True)
        assert all(g['lr'] == a.learning_rate for g in runner.alg.optimizer.param_groups)
        print('BELIEF_FINETUNE_DONE', str(log_dir), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()

"""Basketball belief-state auxiliary head: predict the privileged ball state
from the blind actor's recurrent latent, with a KL anchor to the source
policy (v3 design — the mainstream "post-train with an anchor" recipe).

The head mirrors rsl_rl's RND pattern (module + its own optimizer, living on
the algorithm rather than inside actor/critic):

* rsl_rl's PPO chains actor+critic parameters into ONE optimizer, so head
  parameters must NOT be added to the actor module: the released basketball
  lineage checkpoints restore an Adam state sized for exactly those two
  models, and any extra parameter would break ``optimizer.load_state_dict``.
  Keeping the head algorithm-side leaves ``actor_state_dict`` and the main
  optimizer untouched, so b11/phase-0 checkpoints load strictly and exports
  (``RNNModel.as_onnx`` copies normalizer/rnn/mlp only) are unaffected.
* Labels are the RAW critic ``body_command`` slot (``basketball_state``:
  ball offset *3, ball linear velocity *1, roll rate *0.2). Observation
  normalization happens inside each model on its own group, so the value
  read from ``batch.observations['critic'][..., 55:61]`` is pre-normalizer.

The anchor is a frozen deep copy of the actor taken right after the source
checkpoint loads. Each minibatch pays ``anchor_weight * KL(anchor ‖ current)``
so the belief pressure cannot walk the behavior off the source optimum
(v1 λ=0.5 → 22% survival, v2 λ=0.05+warmup → 85.6%: both unanchored runs
drifted). Anchor forwards reuse the recorded rollout hidden states — the
same approximation PPO itself makes when recomputing log-probabilities.
Continuing an anchored run re-anchors to the LOADED actor (the run's own
latest state), not to the original lineage source.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.utils import unpad_trajectories


class BasketballBeliefPPO(PPO):
    """PPO with an auxiliary ball-state prediction head on the actor latent."""

    def __init__(
        self,
        actor,
        critic,
        storage,
        belief_weight: float = 0.5,
        belief_hidden: int = 64,
        belief_lr: float = 1e-3,
        belief_warmup: int = 100,
        anchor_weight: float = 1.0,
        label_obs: str = "critic",
        label_slice: tuple[int, int] = (55, 61),
        **kwargs,
    ):
        if belief_weight < 0 or anchor_weight < 0:
            raise ValueError("belief_weight and anchor_weight must be non-negative")
        super().__init__(actor, critic, storage, **kwargs)
        if not getattr(actor, "is_recurrent", False):
            raise ValueError("BasketballBeliefPPO requires a recurrent actor")
        self.belief_weight = float(belief_weight)
        self.belief_warmup = int(belief_warmup)
        self.anchor_weight = float(anchor_weight)
        self._anchor_actor = None
        self._update_calls = 0
        self._label_obs = label_obs
        self._label_slice = slice(*label_slice)
        self.belief_head = nn.Sequential(
            nn.Linear(actor.latent_dim, belief_hidden),
            nn.ELU(),
            nn.Linear(belief_hidden, label_slice[1] - label_slice[0]),
        ).to(self.device)
        self.belief_optimizer = torch.optim.Adam(self.belief_head.parameters(), lr=belief_lr)

    def _belief_loss(self, batch, detach_latent: bool = False) -> torch.Tensor:
        """MSE between the head(actor latent) and the raw ball-state labels.

        Both sides go through the same ``unpad_trajectories`` masks call the
        recurrent minibatch generator uses, so rows stay aligned with the
        actor forward pass that ``PPO.update`` performs on the same batch.
        ``detach_latent`` trains the head only (v1 at weight 0.5 wrecked the
        balance policy: the aux gradient into the LSTM crowded out PPO's).
        """
        latent = self.actor.get_latent(
            batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0]
        )
        if detach_latent:
            latent = latent.detach()
        pred = self.belief_head(latent)
        label = unpad_trajectories(batch.observations[self._label_obs], batch.masks)[..., self._label_slice]
        if pred.shape[:-1] != label.shape[:-1]:
            raise RuntimeError(f"belief shapes misaligned: pred {tuple(pred.shape)} label {tuple(label.shape)}")
        return F.mse_loss(pred, label)

    def update(self) -> dict[str, float]:
        """PPO.update with the belief term added (structure mirrors upstream)."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None
        mean_belief_loss = 0
        mean_anchor_kl = 0
        num_anchor_updates = 0

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # KL anchor to the frozen source policy: penalize the current
            # distribution wherever it drops mass the anchor still assigns.
            anchor_kl = None
            if self._anchor_actor is not None and self.anchor_weight > 0:
                with torch.no_grad():
                    # stochastic_output=True only so the distribution gets its
                    # update() call (output_distribution_params requires it);
                    # the sampled actions are discarded.
                    self._anchor_actor(
                        batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0],
                        stochastic_output=True,
                    )
                    anchor_params = tuple(p[:original_batch_size] for p in self._anchor_actor.output_distribution_params)
                anchor_kl = self.actor.get_kl_divergence(
                    anchor_params, distribution_params
                ).mean()

            # Belief auxiliary loss on the same minibatch. During warmup the
            # head trains on a detached latent (own optimizer, separate graph)
            # so the actor is untouched until the head can actually read it.
            warmup = self._update_calls < self.belief_warmup
            belief_loss = self._belief_loss(batch, detach_latent=warmup) if self.belief_weight > 0 else None

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            if belief_loss is not None and not warmup:
                loss = loss + self.belief_weight * belief_loss
            if anchor_kl is not None:
                loss = loss + self.anchor_weight * anchor_kl

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                mean_actions = self.actor(batch.observations.detach().clone())

                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # One combined backward: the belief term rides in the same graph, so
            # the head and the actor each get their gradients from a single pass
            # (a second backward after optimizer.step() would trip autograd's
            # version counters on the in-place-updated parameters).
            self.optimizer.zero_grad()
            if belief_loss is not None:
                self.belief_optimizer.zero_grad()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if belief_loss is not None:
                if warmup:
                    # Detached-latent graph is independent of the actor's, so a
                    # late backward here cannot trip autograd version checks.
                    self.belief_optimizer.zero_grad()
                    belief_loss.backward()
                nn.utils.clip_grad_norm_(self.belief_head.parameters(), self.max_grad_norm)
                self.belief_optimizer.step()
                mean_belief_loss += belief_loss.item()

            if self.rnd:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if anchor_kl is not None:
                mean_anchor_kl += anchor_kl.item()
                num_anchor_updates += 1
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        mean_belief_loss /= num_updates
        if num_anchor_updates:
            mean_anchor_kl /= num_anchor_updates
        # One line per update: visible in stdout smoke logs without a wandb UI.
        phase = "warmup" if self._update_calls < self.belief_warmup else "coupled"
        print(f"BELIEF_LOSS {mean_belief_loss:.6f} ANCHOR_KL {mean_anchor_kl:.6f} phase={phase}", flush=True)
        self._update_calls += 1

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "belief": mean_belief_loss,
        }
        if num_anchor_updates:
            loss_dict["anchor_kl"] = mean_anchor_kl
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["belief_state_dict"] = self.belief_head.state_dict()
        saved_dict["belief_optimizer_state_dict"] = self.belief_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        # Old (pre-belief) checkpoints carry no head state: start from a fresh
        # head. Checkpoints saved by this class restore both head and moments.
        belief_state = loaded_dict.pop("belief_state_dict", None)
        belief_opt = loaded_dict.pop("belief_optimizer_state_dict", None)
        result = super().load(loaded_dict, load_cfg, strict)
        if belief_state is not None:
            self.belief_head.load_state_dict(belief_state)
            if belief_opt is not None:
                self.belief_optimizer.load_state_dict(belief_opt)
        # Snapshot the loaded actor as the frozen anchor BEFORE any training
        # moves it (guarded by the same load_cfg flag that restored it).
        if self.anchor_weight > 0 and (load_cfg is None or load_cfg.get("actor")):
            self._anchor_actor = copy.deepcopy(self.actor)
            self._anchor_actor.eval()
            for p in self._anchor_actor.parameters():
                p.requires_grad_(False)
        return result

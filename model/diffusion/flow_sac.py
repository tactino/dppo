import copy
import logging

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from model.diffusion.flow import FlowModel, Sample

log = logging.getLogger(__name__)


class SACFlow(FlowModel):
    learn_eta = False

    def __init__(
        self,
        actor,
        critic_1,
        critic_2,
        ft_denoising_steps,
        alpha=0.2,
        network_path=None,
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        **kwargs,
    ):
        super().__init__(
            network=actor,
            network_path=network_path,
            **kwargs,
        )
        assert ft_denoising_steps <= self.denoising_steps

        self.ft_denoising_steps = ft_denoising_steps
        self.ft_denoising_steps_cnt = 0

        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std

        self.actor = self.network
        self.actor_ft = copy.deepcopy(self.actor)
        log.info("Cloned model for fine-tuning")

        for param in self.actor.parameters():
            param.requires_grad = False

        log.info(
            f"Number of finetuned parameters: {sum(p.numel() for p in self.actor_ft.parameters() if p.requires_grad)}"
        )

        self.critic_1 = critic_1.to(self.device)
        self.critic_2 = critic_2.to(self.device)

        self.alpha = alpha  # Entropy coefficient

    def step(self):
        self.ft_denoising_steps_cnt += 1

    def get_min_sampling_denoising_std(self):
        if isinstance(self.min_sampling_denoising_std, float):
            return self.min_sampling_denoising_std
        return self.min_sampling_denoising_std()

    def flow(self, x, t, cond, use_base_policy=False):
        vel = self.actor(x, t, cond=cond)
        ft_indices = torch.where(t > 1 - self.ft_denoising_steps / self.denoising_steps)[0]
        actor = self.actor if use_base_policy else self.actor_ft

        if len(ft_indices) > 0:
            cond_ft = {k: cond[k][ft_indices] for k in cond}
            vel_ft = actor(x[ft_indices], t[ft_indices], cond=cond_ft)
            vel[ft_indices] = vel_ft

        return vel

    @torch.no_grad()
    def forward(self, cond, deterministic=False, return_chain=True, use_base_policy=False):
        device = self.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)

        min_sampling_denoising_std = self.get_min_sampling_denoising_std()

        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        dt = 1 / self.denoising_steps
        t_all = [dt * i for i in range(self.denoising_steps)]
        chain = [] if return_chain else None

        if return_chain and self.ft_denoising_steps == self.denoising_steps:
            chain.append(x)

        for i, t in enumerate(t_all):
            t_b = torch.full((B,), t, device=device, dtype=x.dtype)
            vel = self.flow(x, t_b, cond, use_base_policy)

            std = torch.zeros_like(vel) if deterministic else torch.full_like(vel, min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(-self.randn_clip_value, self.randn_clip_value)
            x = x + vel * dt + std * noise

            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(x, -self.final_action_clip_value, self.final_action_clip_value)

            if return_chain and i >= self.denoising_steps - self.ft_denoising_steps - 1:
                chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, chain)

    def update(self, batch):
        # Unpack batch
        obs, actions, next_obs, rewards, dones = batch
        cond = {"state": obs}

        # Critic update
        with torch.no_grad():
            next_sample = self.forward({"state": next_obs}, deterministic=False)
            next_actions = next_sample.trajectories
            next_q1 = self.critic_1(next_obs, next_actions)
            next_q2 = self.critic_2(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * self._approx_entropy(next_actions)
            target_q = rewards + (1 - dones) * next_q

        current_q1 = self.critic_1(obs, actions)
        current_q2 = self.critic_2(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Actor update
        sample = self.forward({"state": obs}, deterministic=False)
        new_actions = sample.trajectories
        q1 = self.critic_1(obs, new_actions)
        q2 = self.critic_2(obs, new_actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * self._approx_entropy(new_actions) - q).mean()

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }

    def _approx_entropy(self, actions):
        # Naive entropy approximation assuming Gaussian noise
        std = torch.full_like(actions, self.min_logprob_denoising_std)
        dist = Normal(actions, std)
        return dist.entropy().mean(dim=-1)

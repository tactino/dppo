"""
DGRPO: Diffusion Grouped Relative Policy Optimization. 

K: number of denoising steps
To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

B: batch size

"""

from typing import Optional
import torch
import torch.nn as nn
import logging
import math

log = logging.getLogger(__name__)
from model.diffusion.diffusion_vpg import VPGDiffusion

class FakeCritic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.tensor(0.0, device=kwargs.get("device", "cpu"))

    def to(self, device):
        return super().to(device)

class GRPODiffusion(VPGDiffusion):
    def __init__(
        self,
        gamma_denoising: float,
        clip_ploss_coef: float,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3,
        clip_advantage_lower_quantile: float = 0, 
        clip_advantage_upper_quantile: float = 1, 
        norm_adv: bool = True,
        grpo_group_size: int = 32,
        **kwargs,
    ):
        fake_critic = FakeCritic()
        super().__init__(
            critic=fake_critic,
            **kwargs
        )

        # Whether to normalize advantages within batch
        self.norm_adv = norm_adv

        # Clipping value for policy loss
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate

        # Discount factor for diffusion MDP
        self.gamma_denoising = gamma_denoising

        # Quantiles for clipping advantages
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile

        self.grpo_group_size = grpo_group_size  

    def loss(
        self,
        obs,
        chains_prev,
        chains_next,
        denoising_inds,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        reward_horizon=4,
    ):
        """
        GRPO loss

        obs: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        chains: (B, K+1, Ta, Da)
        advantages: (B,)
        oldlogprobs: (B, K, Ta, Da)
        use_bc_loss: whether to add BC regularization loss
        """
        # Get new logprobs for denoising steps
        newlogprobs, eta = self.get_logprobs_subsample(
            obs,
            chains_prev,
            chains_next,
            denoising_inds,
            get_ent=True,
        )
        entropy_loss = -eta.mean()
        newlogprobs = newlogprobs.clamp(min=-5, max=2)
        oldlogprobs = oldlogprobs.clamp(min=-5, max=2)

        # only backpropagate through the earlier steps (e.g., ones actually executed in the environment)
        newlogprobs = newlogprobs[:, :reward_horizon, :]
        oldlogprobs = oldlogprobs[:, :reward_horizon, :]

        # Average over action dimensions
        newlogprobs = newlogprobs.mean(dim=(-1, -2)).view(-1)
        oldlogprobs = oldlogprobs.mean(dim=(-1, -2)).view(-1)

        # Behavior cloning loss
        bc_loss = 0
        if use_bc_loss:
            samples = self.forward(
                cond=obs,
                deterministic=False,
                return_chain=True,
                use_base_policy=True,
            )
            bc_logprobs = self.get_logprobs(
                obs,
                samples.chains,
                get_ent=False,
                use_base_policy=False,
            )
            bc_logprobs = bc_logprobs.clamp(min=-5, max=2)
            bc_logprobs = bc_logprobs.mean(dim=(-1, -2)).view(-1)
            bc_loss = -bc_logprobs.mean()

        # Quantile clipping
        adv_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=adv_min, max=adv_max)

        # Denoising discount
        discount = torch.tensor(
            [self.gamma_denoising ** (self.ft_denoising_steps - i - 1) for i in denoising_inds]
        ).to(self.device)
        advantages *= discount

        # Log ratio and ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        # Clipping coefficient
        t = (denoising_inds.float() / (self.ft_denoising_steps - 1)).to(self.device)
        if self.ft_denoising_steps > 1:
            clip_ploss_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (torch.exp(self.clip_ploss_coef_rate * t) - 1) / (math.exp(self.clip_ploss_coef_rate) - 1)
        else:
            clip_ploss_coef = t

        # --- GRPO grouping for pg_loss ---
        B = advantages.shape[0]
        G = self.grpo_group_size
        num_groups = B // G
        remainder = B % G


        total_loss = 0.0

        for i in range(num_groups):
            start = i * G
            end = start + G
            adv_group = advantages[start:end]
            ratio_group = ratio[start:end]
            clip_group = clip_ploss_coef[start:end] if clip_ploss_coef.ndim > 0 else clip_ploss_coef

            # # Permute for better mixing
            # perm = torch.randperm(B, device=advantages.device)
            # advantages = advantages[perm]
            # ratio = ratio[perm]
            # clip_coef = clip_coef[perm]

            adv_group = adv_group - adv_group.mean()

            pg1 = -adv_group * ratio_group
            pg2 = -adv_group * torch.clamp(ratio_group, 1 - clip_group, 1 + clip_group)
            total_loss += torch.max(pg1, pg2).sum()

        # Handle remainder
        if remainder > 0:
            adv_group = advantages[-remainder:]
            ratio_group = ratio[-remainder:]
            clip_group = clip_ploss_coef[-remainder:] if clip_ploss_coef.ndim > 0 else clip_ploss_coef

            adv_group = adv_group - adv_group.mean()

            pg1 = -adv_group * ratio_group
            pg2 = -adv_group * torch.clamp(ratio_group, 1 - clip_group, 1 + clip_group)
            total_loss += torch.max(pg1, pg2).sum()

        pg_loss = total_loss / B

        return (
            pg_loss,
            entropy_loss,
            ratio.mean().item(),
            bc_loss,
            eta.mean().item(),
        )
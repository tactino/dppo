import logging
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)

Sample = namedtuple("Sample", "trajectories chains")

class FlowModel(nn.Module):
    def __init__(
        self,
        network,
        horizon_steps,
        obs_dim,
        action_dim,
        network_path=None,
        device="cuda:0",
        final_action_clip_value=None,
        denoising_steps=10,
        
        flow_sig_min=0.001,
        flow_beta_params=(1.5, 1),
        randn_clip_value=10,
        **kwargs,
    ):
        super().__init__()
        for k, v in kwargs.items():
            log.warning(f"Unused argument {k}: {v}")
        
        self.device = device
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = int(denoising_steps)
        
        # Whether to clamp the final sampled action between [-1, 1]
        self.final_action_clip_value = final_action_clip_value
        self.flow_sig_min = flow_sig_min
        
        self.flow_t_max = 1 - self.flow_sig_min
        self.flow_beta_dist = torch.distributions.Beta(*flow_beta_params)
        self.randn_clip_value = randn_clip_value

        self.network = network.to(device)
        
        if network_path is not None:
            checkpoint = torch.load(
                network_path, map_location=device, weights_only=True
            )
            if "ema" in checkpoint:
                self.load_state_dict(checkpoint["ema"], strict=False)
                logging.info("Loaded SL-trained policy from %s", network_path)
            else:
                self.load_state_dict(checkpoint["model"], strict=False)
                logging.info("Loaded RL-trained policy from %s", network_path)
                
        logging.info(
            f"Number of network parameters: {sum(p.numel() for p in self.parameters())}"
        )
        
    def flow(self, x, t, cond, network_override=None):
        if network_override is not None:
            vel = network_override(x, t, cond=cond)
        else:
            vel = self.network(x, t, cond=cond)
        return vel
        
    @torch.no_grad()
    def forward(self, cond, deterministic=True):
        """
        Forward pass for sampling actions. Used in evaluating pre-trained/fine-tuned policy. Not modifying diffusion clipping

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
        """
        
        device = self.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)
        
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)

        # Core Logic for flow
        dt = 1 / self.denoising_steps
        t_all = [dt * i for i in range(self.denoising_steps)]
        for i, t in enumerate(t_all):
            t_b = torch.full((B,), t, device=device, dtype=x.dtype)
            vel = self.flow(
                x=x,
                t=t_b,
                cond=cond,
                # deterministic=deterministic,
            )
            x = x + vel * dt
            
            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )
        return Sample(x, None) 
    
    # ---------- Supervised training ----------#
    
    def psi_t(self, x0, x1, t):
        return (1 - (1 - self.flow_sig_min) * t[..., None, None]) * x0 + t[..., None, None] * x1
    
    def loss(self, x, cond, *args):
        batch_size = len(x)
        
        z = self.flow_beta_dist.sample((batch_size,)).to(self.device)
        t = self.flow_t_max * (1 - z)

        return self.flow_losses(x, cond, t)
    
    def flow_losses(self, x, cond: dict, t):
        x0, x1 = torch.randn_like(x, device=self.device, dtype=t.dtype), x.clone()
        
        psi_t = self.psi_t(x0, x1, t)
        
        v_psi = self.flow(psi_t, t, cond=cond)
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        l2 = (v_psi - d_psi) ** 2
        return l2.mean()
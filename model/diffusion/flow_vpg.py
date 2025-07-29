"""
Policy gradient with flow policy. VPG: vanilla policy gradient

K: number of denoising steps
To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

"""

import copy
import logging

log = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from model.diffusion.flow import FlowModel, Sample

class VPGFlow(FlowModel):
    learn_eta = False
    def __init__(
        self,
        actor,
        critic,
        ft_denoising_steps,
        network_path=None,
        # modifying denoising schedule
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
        
        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std
        
        # Minimum std used in calculating denoising logprobs - for stability
        self.min_logprob_denoising_std = min_logprob_denoising_std
        
        # Re-name network to actor
        self.actor = self.network
        
        # Make a copy of the original model
        self.actor_ft = copy.deepcopy(self.actor)
        logging.info(f"Cloned model for fine-tuning")
        
        # Turn off gradients for original model
        for param in self.actor.parameters():
            param.requires_grad = False
        logging.info("Turned off gradients of the pretrained network")
        logging.info(
            f"Number of finetuned parameters: {sum(p.numel() for p in self.actor_ft.parameters() if p.requires_grad)}"
        )
        
        self.critic = critic.to(self.device)
        # Value function
        self.critic = critic.to(self.device)
        if network_path is not None:
            checkpoint = torch.load(
                network_path, map_location=self.device, weights_only=True
            )
            if "ema" not in checkpoint:  # load trained RL model
                self.load_state_dict(checkpoint["model"], strict=False)
                logging.info("Loaded critic from %s", network_path)
                
    # ---------- Sampling ----------#

    def step(self):
        """
        Anneal min_sampling_denoising_std and fine-tuning denoising steps

        Current configs do not apply annealing
        """
        
        # anneal denoising steps
        self.ft_denoising_steps_cnt += 1
        
        # TODO: add anealing denoising steps
        pass
    
    def get_min_sampling_denoising_std(self):
        if type(self.min_sampling_denoising_std) is float:
            return self.min_sampling_denoising_std
        else:
            return self.min_sampling_denoising_std()
        
    def flow(
        self,
        x,
        t,
        cond,
        use_base_policy=False,
    ):
        vel = self.actor(x, t, cond=cond)
        ft_indices = torch.where(t > 1 - self.ft_denoising_steps / self.denoising_steps)[0]
        
        # Use base policy to query expert model, e.g. for imitation loss
        actor = self.actor if use_base_policy else self.actor_ft
        
        # overwrite the model with the base policy
        if len(ft_indices) > 0:
            cond_ft = {key: cond[key][ft_indices] for key in cond}
            vel_ft = actor(x[ft_indices], t[ft_indices], cond=cond_ft)
            vel[ft_indices] = vel_ft
            
        return vel
    
    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
        return_chain=True,
        use_base_policy=False,
    ):
        """
        Forward pass for sampling actions.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            deterministic: 
            return_chain: whether to return the entire chain of denoised actions
            use_base_policy: whether to use the frozen pre-trained policy instead
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
                chain: (B, K + 1, Ta, Da)
        """
        device = self.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)

        # Get updated minimum sampling denoising std
        min_sampling_denoising_std = self.get_min_sampling_denoising_std()
        
        # Loop
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        dt = 1 / self.denoising_steps
        t_all = [dt * i for i in range(self.denoising_steps)]
        chain = [] if return_chain else None
        
        if return_chain and self.ft_denoising_steps == self.denoising_steps:
            chain.append(x)
            
        for i, t in enumerate(t_all):
            t_b = torch.full((B,), t, device=device, dtype=x.dtype)
            vel = self.flow(
                x=x,
                t=t_b,
                cond=cond,
                use_base_policy=use_base_policy,
            )
            st = torch.zeros_like(vel)
            if deterministic and i == len(t_all) - 1:
                std = torch.zeros_like(vel)
            elif deterministic:
                std = torch.zeros_like(vel)
            else:
                std = torch.clip(st, min=min_sampling_denoising_std)
                
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            
            x = x + vel * dt + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )

            if return_chain:
                if i >= self.denoising_steps - self.ft_denoising_steps - 1:
                    chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, chain)
    
    def get_logprobs(
        self,
        cond,
        chains,
        get_ent: bool = False,
        use_base_policy: bool = False,  
    ):
        """
        Calculating the logprobs of the entire chain of denoised actions.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            get_ent: flag for returning entropy
            use_base_policy: flag for using base policy

        Returns:
            logprobs: (B x K, Ta, Da)
            entropy (if get_ent=True):  (B x K, Ta)
        """
        # Repeat cond for denoising_steps, flatten batch and time dimensions
        cond = {
            key: cond[key]
            .unsqueeze(1)
            .repeat(1, self.ft_denoising_steps, *(1,) * (cond[key].ndim - 1))
            .flatten(start_dim=0, end_dim=1)
            for key in cond
        }  # less memory usage than einops?
        
        t_single = torch.arange(
            start=self.denoising_steps - self.ft_denoising_steps,
            end=self.denoising_steps,
            step=1,
            device=self.device,
        ) / self.denoising_steps
        
        t_all = t_single.repeat(chains.shape[0], 1).flatten()
        
        # Split chains
        chains_prev = chains[:, :-1]
        chains_next = chains[:, 1:]
        
        # Flatten first two dimensions
        chains_prev = chains_prev.reshape(-1, self.horizon_steps, self.action_dim)
        chains_next = chains_next.reshape(-1, self.horizon_steps, self.action_dim)
        
        # Forward pass with previous chains
        vel = self.flow(
            x=chains_prev,
            t=t_all,
            cond=cond,
            use_base_policy=use_base_policy,
        )
        std = torch.full_like(vel, self.min_logprob_denoising_std, device=self.device)
        dist = Normal(chains_prev + vel * (1 / self.denoising_steps), std)
        log_prob = dist.log_prob(chains_next)
        
        if get_ent:
           return log_prob, -dist.entropy()
        return log_prob
   
    def get_logprobs_subsample(
        self,
        cond,
        chains_prev,
        chains_next,
        denoising_inds,
        get_ent: bool = False,
    ):
        """
        Calculating the logprobs of random samples of denoised chains.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            get_ent: flag for returning entropy
            use_base_policy: flag for using base policy

        Returns:
            logprobs: (B, Ta, Da)
            entropy (if get_ent=True):  (B, Ta)
            denoising_indices: (B, )
        """
        
        t_single = torch.arange(
            start=self.denoising_steps - self.ft_denoising_steps,
            end=self.denoising_steps,
            step=1,
            device=self.device,
        ) / self.denoising_steps
        
        t_all = t_single[denoising_inds]
        
        vel = self.flow(
            x=chains_prev,
            t=t_all,
            cond=cond,
            use_base_policy=False,
        )
        std = torch.full_like(vel, self.min_logprob_denoising_std, device=self.device)
        dist = Normal(chains_prev + vel * (1 / self.denoising_steps), std)
        
        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, -dist.entropy()
        return log_prob
"""
For evaluating RL fine-tuned diffusion policy

Account for frozen base policy for early denoising steps and fine-tuned policy for later denoising steps

"""

import copy
import logging

import torch

log = logging.getLogger(__name__)

from model.diffusion.flow import FlowModel

class FlowEval(FlowModel):
    def __init__(
        self,
        network_path,
        ft_denoising_steps,  # if running pre-trained model (not fine-tuned), set it to zero; if running fine-tuned model, need to specify the correct number of denoising steps fine-tuned, so that here it knows which model (base or ft) to use for each denoising step
        **kwargs,
    ):
        # do not let base class load model
        super().__init__(network_path=None, **kwargs)
        self.ft_denoising_steps = ft_denoising_steps
        checkpoint = torch.load(
            network_path, map_location=self.device, weights_only=True
        )  # 'network.mlp_mean...', 'actor.mlp_mean...', 'actor_ft.mlp_mean...'

        # Set up base model --- techncally not needed if all denoising steps are fine-tuned
        self.actor = self.network
        try:
            base_weights = {
                key.split("actor.")[1]: checkpoint["model"][key]
                for key in checkpoint["model"]
                if "actor." in key
            }
            use_ft = True
            self.actor.load_state_dict(base_weights, strict=True)
        except Exception:
            assert ft_denoising_steps == 0, (
                "If no base policy weights are found, ft_denoising_steps must be 0"
            )
            base_weights = {
                key.split("network.")[1]: checkpoint["model"][key]
                for key in checkpoint["model"]
                if "network." in key
            }
            use_ft = False
            logging.info("Actor weights not found. Using pre-trained weights!")
            self.actor.load_state_dict(base_weights, strict=True)
        logging.info("Loaded base policy weights from %s", network_path)

        # Always set up fine-tuned model
        if use_ft:
            self.actor_ft = copy.deepcopy(self.network)
            ft_weights = {
                key.split("actor_ft.")[1]: checkpoint["model"][key]
                for key in checkpoint["model"]
                if "actor_ft." in key
            }
            self.actor_ft.load_state_dict(ft_weights, strict=True)
            logging.info("Loaded fine-tuned policy weights from %s", network_path)
            
    def flow(
        self,
        x,
        t,
        cond,
    ):
        vel = self.actor(x, t, cond=cond)
        ft_indices = torch.where(t > 1 - self.ft_denoising_steps / self.denoising_steps)[0]
        
        # overwrite the model with the base policy
        if len(ft_indices) > 0:
            cond_ft = {key: cond[key][ft_indices] for key in cond}
            vel_ft = self.actor_ft(x[ft_indices], t[ft_indices], cond=cond_ft)
            vel[ft_indices] = vel_ft
            
        return vel
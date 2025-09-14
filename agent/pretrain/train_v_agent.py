"""
Pre-training V network
"""

import logging
import swanlab
import numpy as np
import torch.nn.functional as F

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device


class TrainVAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

    def reset_parameters(self):
        pass

    def to_device(self, batch):
        cond = {}
        for k, v in batch.conditions.items():
            cond[k] = v.to(self.device)
        reward_to_gos = batch.reward_to_gos.to(self.device)
        return cond, reward_to_gos

    def v_loss(self, batch):
        """
        Supervised MC return regression:
        V(s) ≈ reward-to-go
        """
        cond, reward_to_gos = self.to_device(batch)
        target_v = reward_to_gos.view(-1)


        v_pred = self.model(cond)  # 只输入状态

        # 如果是 tuple（unlikely），处理方式同 Q
        if isinstance(v_pred, tuple):
            v1, v2 = v_pred
            loss1 = F.mse_loss(v1, target_v)
            loss2 = F.mse_loss(v2, target_v)
            loss = 0.5 * (loss1 + loss2)
        else:
            loss = F.mse_loss(v_pred, target_v)
        return loss

    def run(self):

        timer = Timer()
        self.epoch = 1

        for _ in range(self.n_epochs):

            # ------------------ Training ------------------ #
            loss_train_epoch = []

            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)

                self.model.train()
                loss_train = loss_train = self.v_loss(batch_train)
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())

                self.optimizer.step()
                self.optimizer.zero_grad()

            loss_train = np.mean(loss_train_epoch)

            # ------------------ Validation ------------------ #
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                for batch_val in self.dataloader_val:
                    if self.dataset_val.device == "cpu":
                        batch_val = batch_to_device(batch_val)
                    loss_val = self.v_loss(batch_val)
                    loss_val_epoch.append(loss_val.item())
                self.model.train()
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # ------------------ Learning rate update ------------------ #
            self.lr_scheduler.step()

            # ------------------ Save model ------------------ #
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # ------------------ Logging ------------------ #
            if self.epoch % self.log_freq == 0:
                log.info(f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}")
                if self.use_swanlab:
                    if loss_val is not None:
                        swanlab.log({"v_loss - val": loss_val}, step=self.epoch, commit=False)
                    swanlab.log({"v_loss - train": loss_train}, step=self.epoch, commit=True)

            self.epoch += 1

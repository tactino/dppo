import logging
import swanlab
import numpy as np
import torch.nn.functional as F

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device


class TrainQAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = self.model.to(self.device)

    def reset_parameters(self):
        pass

    def to_device(self, batch):
        cond = {k: v.to(self.device) for k, v in batch.conditions.items()}
        actions = batch.actions.to(self.device)
        reward_to_gos = batch.reward_to_gos.to(self.device)
        return cond, actions, reward_to_gos

    def q_loss(self, batch):
        """
        Supervised MC return regression:
        Q(s,a) ≈ reward-to-go
        """
        cond, actions, target_q = self.to_device(batch)
        target_q = target_q.squeeze(-1)
        q_pred = self.model(cond, actions)

        if self.epoch == 1 and np.random.rand() < 0.01:
            print("state:", cond["state"].shape)
            print("actions:", actions.shape)
            print("target_q:", target_q.shape)

        if isinstance(q_pred, tuple):
            q1, q2 = q_pred
            loss1 = F.mse_loss(q1, target_q)
            loss2 = F.mse_loss(q2, target_q)
            loss = 0.5 * (loss1 + loss2)
        else:
            loss = F.mse_loss(q_pred, target_q)
        return loss


    def run(self):

        timer = Timer()
        self.epoch = 1

        for _ in range(self.n_epochs):

            # ------------------ Training ------------------ #
            loss_train_epoch = []

            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train, self.device)

                self.model.train()
                loss_train = self.q_loss(batch_train)   # 🔹 改成用自己的 q_loss
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
                        batch_val = batch_to_device(batch_val, self.device)
                    loss_val = self.q_loss(batch_val)   # 🔹 改成用自己的 q_loss
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
                        swanlab.log({"q_loss - val": loss_val}, step=self.epoch, commit=False)
                    swanlab.log({"q_loss - train": loss_train}, step=self.epoch, commit=True)

            self.epoch += 1

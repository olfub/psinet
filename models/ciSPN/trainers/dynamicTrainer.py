import math

import torch
from rtpt import RTPT

from models.ciSPN.environment import environment


class DynamicTrainer:
    def __init__(
        self,
        model,
        conf,
        loss,
        train_loss=False,
        lr=1e-3,
        pre_epoch_callback=None,
        optimizer="adam",
        scheduler_fn=None,
    ):
        self.model = model
        self.conf = conf
        self.base_lr = lr

        self.loss = loss

        trainable_params = [{"params": model.parameters(), "lr": lr}]
        if train_loss:
            loss.train()
            trainable_params.append({"params": loss.parameters(), "lr": lr})
        else:
            loss.eval()

        print(f"Optimizer: {optimizer}")
        print("Learning Rate: {}".format(lr))

        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(trainable_params)  # , amsgrad=True)
            self.scheduler = None
        elif optimizer == "adamWD":
            self.optimizer = torch.optim.Adam(trainable_params, weight_decay=0.01)
            self.scheduler = None
        elif optimizer == "adamAMS":
            self.optimizer = torch.optim.Adam(trainable_params, amsgrad=True)
            self.scheduler = None
        elif optimizer == "adamW":
            self.optimizer = torch.optim.AdamW(trainable_params)  # , amsgrad=True)
            self.scheduler = None
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(trainable_params, lr, momentum=0.9)
            self.scheduler = (
                scheduler_fn(self.optimizer) if scheduler_fn is not None else None
            )
        elif optimizer == "sgdWD":
            self.optimizer = torch.optim.SGD(
                trainable_params, lr, momentum=0.9, weight_decay=0.01
            )
            self.scheduler = (
                scheduler_fn(self.optimizer) if scheduler_fn is not None else None
            )
        else:
            raise ValueError(f"unknown optimizer {optimizer}")

        self.pre_epoch_callback = pre_epoch_callback

    def run_training(self, provider):
        loss_curve = []
        loss_value = -math.inf

        # set spn into training mode
        if torch.cuda.is_available():
            self.model.cuda().train()
        else:
            self.model.train()

        loss_name = self.loss.get_name()

        rtpt = RTPT(
            name_initials=environment["runtime"]["initials"],
            experiment_name="cfSPN Training",
            max_iterations=self.conf.num_epochs,
        )
        rtpt.start()

        for epoch in range(self.conf.num_epochs):
            if self.pre_epoch_callback is not None:
                self.pre_epoch_callback(epoch, loss_value, self.loss)

            batch_num = 0
            while provider.has_data():
                x, y = provider.get_next_batch()

                self.optimizer.zero_grad(set_to_none=True)
                self.loss.zero_grad(set_to_none=True)

                prediction = self.model.forward(x, y)
                cur_loss = self.loss.forward(x, y, prediction)

                cur_loss.backward()
                self.optimizer.step()

                if batch_num % 100 == 0:
                    cur_loss_np = cur_loss.cpu().item()
                    print(
                        f"ep. {epoch}, batch {batch_num}, train {loss_name} {cur_loss_np:.2f}",
                        end="\r",
                        flush=True,
                    )
                batch_num += 1

            rtpt.step(f"ep{epoch}/{self.conf.num_epochs}")

            loss_value = cur_loss.detach().cpu().item()
            loss_curve.append(loss_value)

            provider.reset()

            if self.scheduler is not None:
                self.scheduler.step()

        return loss_curve

    def run_training_dataloader(self, dataloader, batch_processor, gradient_clipping=0):
        loss_curve = []
        loss_value = -math.inf

        # set spn into training mode
        if torch.cuda.is_available():
            self.model.cuda().train()
        else:
            self.model.train()

        loss_name = self.loss.get_name()

        rtpt = RTPT(
            name_initials=environment["runtime"]["initials"],
            experiment_name="cfSPN Training",
            max_iterations=self.conf.num_epochs,
        )
        rtpt.start()

        for epoch in range(self.conf.num_epochs):
            if self.pre_epoch_callback is not None:
                self.pre_epoch_callback(epoch, loss_value, self.loss)

            for batch_num, batch in enumerate(dataloader):
                x, y = batch_processor(batch)

                self.optimizer.zero_grad(set_to_none=True)
                self.loss.zero_grad(set_to_none=True)

                prediction = self.model.forward(x, y)
                cur_loss = self.loss.forward(x, y, prediction)

                cur_loss.backward()

                if gradient_clipping != 0:
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), gradient_clipping
                    )

                self.optimizer.step()

                if batch_num % 100 == 0:
                    cur_loss_np = cur_loss.cpu().item()
                    print(
                        f"ep. {epoch}, batch {batch_num}, train {loss_name} {cur_loss_np:.2f}",
                        end="\r",
                        flush=True,
                    )

            rtpt.step()  # f"ep{epoch}/{self.conf.num_epochs}")

            loss_value = cur_loss.detach().cpu().item()
            loss_curve.append(loss_value)

            if self.scheduler is not None:
                self.scheduler.step()

        return loss_curve

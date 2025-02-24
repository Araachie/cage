import os
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_polynomial_decay_schedule_with_warmup

from lutils.configuration import Configuration
from lutils.constants import MAIN_PROCESS
from lutils.dict_wrapper import DictWrapper
from lutils.logger import Logger
from training.utils import check_ddp_consistency


class Trainer:
    """
    Class that handles the training
    """

    def __init__(
            self,
            rank: int,
            run_name: str,
            config: Configuration,
            dataset: Dataset,
            sampler: torch.utils.data.distributed.Sampler,
            num_gpus: int,
            device: torch.device):
        """
        Initializes the Trainer

        :param rank: rank of the current process
        :param config: training configuration
        :param dataset: dataset to train on
        :param sampler: sampler to create the dataloader with
        :param device: device to use for training
        """
        super(Trainer, self).__init__()

        self.config = config
        self.rank = rank
        self.is_main_process = self.rank == MAIN_PROCESS
        self.num_gpus = num_gpus
        self.device = device

        # Create folder for saving
        self.run_path = os.path.join("runs", run_name)
        os.makedirs(self.run_path, exist_ok=True)
        os.makedirs(os.path.join(self.run_path, "checkpoints"), exist_ok=True)

        # Setup dataloader
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config["batching"]["batch_size"],
            shuffle=False,
            num_workers=self.config["batching"]["num_workers"],
            sampler=sampler,
            pin_memory=True)

        # Optimizer will be defined in train_epoch
        self.optimizer = None

        # Scheduler will be defined in train_epoch
        self.lr_scheduler = None

        self.global_step = 0

    def init_optimizer(self, model: nn.Module):
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["optimizer"]["learning_rate"],
            weight_decay=self.config["optimizer"]["weight_decay"],
            betas=(0.9, 0.999))
        self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config["optimizer"]["num_warmup_steps"],
            num_training_steps=self.config["optimizer"]["num_training_steps"],
            power=0.5)

    def get_lr(self):
        assert self.optimizer is not None

        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(
            self,
            model: nn.Module,
            logger: Logger,
            scalar_logging_frequency: int = 100,
            saving_frequency: int = 5000,
            checkpointing_frequency: int = 20000):
        """
        Trains the model for one epoch

        """

        model.train()

        # Setup optimizer and scheduler if not yet
        if self.optimizer is None:
            self.init_optimizer(model)

        scaler = torch.cuda.amp.GradScaler()

        # Setup loading bar
        train_gen = tqdm(self.dataloader, desc="Batches", disable=not self.is_main_process, leave=False)
        for batch in train_gen:
            # Fetch data
            observations = batch.cuda()
            num_observations = self.config["num_observations"]
            observations = observations[:, :num_observations]

            # Zero gradients
            model.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward the model
                model_outputs = model(observations)

                # Compute the loss
                loss, auxiliary_output = self.calculate_loss(model_outputs)

            # Backward pass
            scaler.scale(loss).backward()

            # Optimizer step
            if self.global_step > self.config["optimizer"]["num_warmup_steps"]:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(self.optimizer)
            self.lr_scheduler.step()

            # Update scaler for the next iteration
            scaler.update()

            # Log scalars
            if self.global_step % scalar_logging_frequency == 0 and self.is_main_process:
                self.log_scalars(
                    loss_terms=auxiliary_output,
                    other_data=DictWrapper(
                        num_controls=model_outputs.num_controls.item()),
                    model=model,
                    log_model_stats=False,
                    logger=logger)

            # Finalize logs
            logger.finalize_logs(step=self.global_step)

            # Save checkpoint
            if self.global_step % checkpointing_frequency == 0:
                self.save_checkpoint(model, f"step_{self.global_step}.pth")
            elif self.global_step % saving_frequency == 0:
                self.save_checkpoint(model)

            self.global_step += 1

        # Close loading bar
        train_gen.close()

        # Save the model
        logger.info("Saving the trained model...")
        self.save_checkpoint(model, f"final_step_{self.global_step}.pth")

    def calculate_loss(
            self,
            results: DictWrapper[str, Any]) -> Tuple[torch.Tensor, DictWrapper[str, Any]]:
        """
        Calculates the loss

        :param results: Dict with the model outputs
        :return: [1,] The loss value, dict with aux outputs
        """

        # Masked flow matching loss
        num_channels = results.reconstructed_vectors.size(2)
        flow_matching_loss = torch.pow(
            results.reconstructed_vectors - results.target_vectors.detach(), 2)  # [b l c h w]
        flow_matching_loss = (flow_matching_loss * results.selected_masks).sum()  # []
        flow_matching_loss = flow_matching_loss / (num_channels * results.selected_masks.sum())  # []

        # Sum up all the losses
        loss_weights = self.config["loss_weights"]
        loss = \
            loss_weights["flow_matching_loss"] * flow_matching_loss

        # DDP hack
        def add_zero_to_loss(value):
            if v is None:
                return loss
            return loss + value.mul(0).mean()

        for _, v in results.items():
            if isinstance(v, list):
                for ev in v:
                    loss = add_zero_to_loss(ev)
            else:
                loss = add_zero_to_loss(v)

        # Create auxiliary output
        auxiliary_output = DictWrapper(
            # Total loss
            total_loss=loss,

            # Loss terms
            flow_matching_loss=flow_matching_loss
        )

        return loss, auxiliary_output

    def log_scalars(
            self,
            loss_terms: DictWrapper[str, Any],
            other_data: DictWrapper[str, Any],
            model: nn.Module,
            log_model_stats: bool,
            logger: Logger):
        for k, v in loss_terms.items():
            logger.log(f"Training/Loss/{k}", v)

        # Log training stats
        logger.log(f"Training/Stats/learning_rate", self.get_lr())
        logger.log(f"Training/Stats/total_loss_is_nan", torch.isnan(loss_terms.total_loss).to(torch.int8))
        logger.log(f"Training/Stats/total_loss_is_inf", torch.isinf(loss_terms.total_loss).to(torch.int8))

        # Other stats
        for k, v in other_data.items():
            logger.log(f"Training/Stats/{k}", v)

        # Model params stats
        if log_model_stats:
            for name, param in model.named_parameters():
                if "vector_field_regressor" in name:
                    logger.log(f"Training/Weights/{name}_norm", param.norm(2))
                    if param.grad is not None:
                        logger.log(f"Training/Weights/{name}_grad_norm", param.grad.norm(2))

    def save_checkpoint(self, model: nn.Module, checkpoint_name: str = None):
        if self.num_gpus > 1:
            check_ddp_consistency(model, r".*\..+_(mean|var|tracked)")

        if self.is_main_process:
            state_dict = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "model": model.state_dict(),
                "global_step": self.global_step
            }
            if checkpoint_name:
                torch.save(state_dict, os.path.join(self.run_path, "checkpoints", checkpoint_name))
            torch.save(state_dict, os.path.join(self.run_path, "checkpoints", "latest.pth"))

    def load_checkpoint(self, model: nn.Module, checkpoint_name: str = None):
        if checkpoint_name is None:
            checkpoint_name = "latest.pth"
        filename = os.path.join(self.run_path, "checkpoints", checkpoint_name)
        if not os.path.isfile(filename):
            raise Exception(f"Cannot load model: no checkpoint found at '{filename}'")

        # Init optimizer and scheduler if not yet
        if self.optimizer is None:
            self.init_optimizer(model)

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        loaded_state = torch.load(filename, map_location=map_location)
        self.optimizer.load_state_dict(loaded_state["optimizer"])
        self.lr_scheduler.load_state_dict(loaded_state["lr_scheduler"])

        is_ddp = False
        for k in loaded_state["model"]:
            if k.startswith("module"):
                is_ddp = True
                break
        if is_ddp:
            state = {k.replace("module.", ""): v for k, v in loaded_state["model"].items()}
        else:
            state = {f"module.{k}": v for k, v in loaded_state["model"].items()}

        dmodel = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        dmodel.load_state_dict(state)

        self.global_step = loaded_state["global_step"]

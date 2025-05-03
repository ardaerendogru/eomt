# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from PyTorch Lightning,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------


from types import MethodType
from gitignore_parser import parse_gitignore
import logging
import torch
import warnings
from lightning.pytorch import cli
from lightning.pytorch.callbacks import ModelSummary, LearningRateMonitor
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
import yaml
from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule
import os
os.environ["NCCL_P2P_DISABLE"] = "1"

def _should_check_val_fx(self: _TrainingEpochLoop, data_fetcher: _DataFetcher) -> bool:
    if not self._should_check_val_epoch():
        return False

    # Use val_check_interval instead of val_check_batch for compatibility with newer Lightning CLI
    val_check_interval = self.trainer.val_check_interval if hasattr(self.trainer, 'val_check_interval') else None

    is_infinite_dataset = val_check_interval is None # Treat absence of interval like infinite check
    is_last_batch = self.batch_progress.is_last_batch
    if is_last_batch and (
        is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
    ):
        return True

    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True

    is_val_check_step = is_last_batch
    if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
        is_val_check_step = (
            self.batch_idx + 1
        ) % self.trainer.limit_train_batches == 0
    elif val_check_interval is not None and val_check_interval > 0:
        # Check validation based on global steps (iterations)
        is_val_check_step = (
            self.global_step % val_check_interval == 0 and not self._should_accumulate()
        )

    return is_val_check_step


class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        warnings.filterwarnings(
            "ignore",
            message=r".*It is recommended to use .* when logging on epoch level in distributed setting to accumulate the metric across devices.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"^The ``compute`` method of metric PanopticQuality was called before the ``update`` method.*",
        )
        warnings.filterwarnings(
            "ignore", message=r"^Grad strides do not match bucket view strides.*"
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Graph break due to unsupported builtin PIL\._imaging\.fill.*",
        )

        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile_disabled", action="store_true")

        parser.link_arguments(
            "data.init_args.num_classes", "model.init_args.num_classes"
        )
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.network.init_args.num_classes",
        )

        parser.link_arguments(
            "data.init_args.stuff_classes", "model.init_args.stuff_classes"
        )

        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments(
            "data.init_args.img_size", "model.init_args.network.init_args.img_size"
        )
        parser.link_arguments(
            "data.init_args.img_size",
            "model.init_args.network.init_args.encoder.init_args.img_size",
        )
 


    def fit(self, model, **kwargs):
        if hasattr(self.trainer.logger.experiment, "log_code"):
            is_gitignored = parse_gitignore(".gitignore")
            include_fn = lambda path: path.endswith(".py") or path.endswith(".yaml")
            self.trainer.logger.experiment.log_code(
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )



        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )

        if not self.config[self.config["subcommand"]]["compile_disabled"]:
            model = torch.compile(model)

        self.trainer.fit(model, **kwargs)


def cli_main():
    LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "enable_model_summary": False,
            "callbacks": [
                ModelSummary(max_depth=3),
                LearningRateMonitor(logging_interval="step"),
            ],
            "devices": 1,
            "gradient_clip_val": 0.01,
            "gradient_clip_algorithm": "norm",
            "accumulate_grad_batches": 1,
            "check_val_every_n_epoch": None,
            "max_epochs": -1,
        },
    )


if __name__ == "__main__":
    cli_main()

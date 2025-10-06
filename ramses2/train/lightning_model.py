import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn.functional as F
from ..model.model import RAMSESModel
from ..train.loss import compute_image_loss
from pathlib import Path
import os
import json
from ..utils.utils import compute_cls_targets
from .metrics import Precision, Recall


class TrainConfig:

    def __init__(self, **kwargs):
        # defaults parameters
        self.losses = {"cls": True, "seg": True, "mass": True}
        self.seg_loss = "dice"
        self.label_smoothing = 0
        self.cls_threshold = 0.5
        self.max_pos = 768
        self.mask_quality_weighting = True

        # Update defaults parameters with kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):

        s = ""

        for k, v in self.__dict__.items():
            s += "{}:{}\n".format(k, v)

        return s

    def save(self, filename):

        # data = {k:v for k, v in self.__dict__.items()}

        p = Path(filename).parent.absolute()
        if not os.path.isdir(p):
            os.mkdir(p)

        with open(filename, "w") as f:
            json.dump(self.__dict__, f)


default_scheduler_config = [
    {
        "name": "ReduceLROnPlateau",
        "mode": "min",
        "patience": 4,
        "factor": 0.5,
        "monitor": "train_loss",
        "interval": "epoch",
        "frequency": 1,
    },
]

default_opt_config = {"optimizer": "AdamW", "lr": 1e-3}


class RAMSESLightning(pl.LightningModule):
    """
    PyTorch Lightning module for training and validating the RAMSES model.

    This class encapsulates the model, optimizer, scheduler, and training/validation logic for the RAMSES architecture.
    It supports flexible configuration of losses, learning rate schedulers, and optimizer parameters via config objects.

    Args:
        config: Configuration object containing model architecture parameters (e.g., strides, grid_sizes, ncls, etc.).
        train_config: Configuration object containing training parameters (e.g., learning rate, loss types, thresholds, etc.).
        optimizer_config (dict, optional): Dictionary of optimizer parameters (type, learning rate, etc.).
        scheduler_config (list of dict, optional): List of scheduler configuration dictionaries, each specifying the scheduler name, parameters, and Lightning-specific options (interval, monitor, frequency, etc.).

    Attributes:
        model: The RAMSESModel instance.
        learning_rate: Learning rate for the optimizer.
        strides: Model stride values.
        grid_sizes: Model grid sizes.
        ncls: Number of classes (excluding background).
        optimizer: an optimizer (already configured with lr and parameters)
        scheduler_config: List of scheduler configuration dictionaries.
        config: Model configuration object.
        losses: Dictionary specifying which losses to compute ("cls", "seg", "mass").
        seg_loss: Segmentation loss function name.
        label_smoothing: Label smoothing factor for classification loss.
        cls_threshold: Threshold for class prediction (used in metrics).

    Methods:
        forward(x, training=True): Forward pass through the model.
        training_step(batch, batch_idx): Training step for a single batch, computes losses and metrics.
        validation_step(batch, batch_idx): Validation step for a single batch, computes losses and metrics.
        configure_optimizers(): Configures optimizer and learning rate schedulers for Lightning.
    """

    def __init__(
        self,
        model,
        train_config,
        optimizer,
        scheduler_config=None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = train_config.lr
        self.strides = model.config.strides
        self.grid_sizes = model.config.grid_sizes
        self.ncls = model.config.ncls
        self.opt = optimizer
        self.scheduler_config = scheduler_config
        self.config = model.config
        self.losses = train_config.losses
        self.max_pos = train_config.max_pos
        self.seg_loss = train_config.seg_loss
        self.label_smoothing = train_config.label_smoothing
        self.cls_threshold = train_config.cls_threshold
        self.mask_quality_weighting = train_config.mask_quality_weighting

    def forward(self, x, training=True):
        return self.model(x, training=training)

    def training_step(self, batch, batch_idx):
        # batch: dict with keys 'image', 'masks', 'category_id', 'label', 'mass'
        images = batch["image"]
        masks = batch["masks"]
        labels = batch["label"]
        category_id = batch["category_id"]
        mass = batch["mass"]
        # print(batch["filename"])
        # Forward pass
        flat_pred_cls, flat_cls_factor, flat_pred_kernel, seg_preds, geom_features = self.model(images, training=True)
        flat_pred_cls = flat_pred_cls.permute(0, 2, 1)  # [B, ncls, nloc] -> [B, nloc, ncls]

        # Pour chaque élément du batch, calculer la loss individuellement
        total_cls_loss, total_seg_loss, total_density_loss = 0, 0, 0
        batch_size = images.shape[0]
        recalls = []
        precisions = []

        for i in range(batch_size):
            # compute targets
            class_targets, label_targets, mass_targets = compute_cls_targets(
                category_id[i],
                masks[i],
                labels[i],
                mass[i],
                images[i].shape[-2:],
                strides=self.strides,
                grid_sizes=self.grid_sizes,
                scale_ranges=self.config.scale_ranges,
                mode="diag",
                offset_factor=self.config.offset_factor,
            )
            # one hot encoding. Delete the bg slice
            class_targets = F.one_hot(class_targets.long(), self.ncls + 1)[..., 1:]
            # Flattening
            # class_targets = class_targets.view(-1, self.ncls)
            # label_targets = label_targets.view(-1)
            # mass_targets = mass_targets.view(-1)
            # Prépare les inputs pour compute_image_loss (voir docstring dans loss.py)

            inputs = (
                class_targets,  # [nloc, n_classes]
                label_targets,  # [nloc]
                mass_targets,  # [nloc]
                masks[i],  # [H, W]
                flat_pred_cls[i],  # [nloc, n_classes]
                flat_cls_factor[i].squeeze(0),  # [nloc]
                flat_pred_kernel[i].T,  # [nloc, k*k*in_ch]
                seg_preds[i],  # [in_ch, H, W]
                geom_features[i],  # [1, H, W]
            )
            cls_loss, seg_loss, density_loss = compute_image_loss(
                inputs,
                compute_cls_loss=self.losses["cls"],
                compute_seg_loss=self.losses["seg"],
                compute_density_loss=self.losses["mass"],
                seg_loss_func=self.seg_loss,
                max_pos=self.max_pos,
                label_smoothing=self.label_smoothing,
                mask_quality_weighting=self.mask_quality_weighting,
            )
            total_cls_loss += cls_loss
            total_seg_loss += seg_loss
            total_density_loss += density_loss

            # metrics
            recall = Recall(class_targets, flat_pred_cls[i])
            precision = Precision(class_targets, flat_pred_cls[i])
            recalls.append(recall)
            precisions.append(precision)

        # Moyenne sur le batch
        total_cls_loss /= batch_size
        total_seg_loss /= batch_size
        total_density_loss /= batch_size
        total_loss = total_cls_loss + total_seg_loss + total_density_loss
        mean_recall = sum(recalls) / len(recalls)
        mean_precision = sum(precisions) / len(precisions)

        self.log("train_precision", mean_precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_recall", mean_recall, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.losses["cls"]:
            self.log("train_cls_loss", total_cls_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.losses["seg"]:
            self.log("train_seg_loss", total_seg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.losses["mass"]:
            self.log("train_density_loss", total_density_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def configure_optimizers(self):
        # Optimizer configurable via self.config.optimizer (dict)
        # Note that we pass only the model's parameters with require_grad=True
        # opt_config = dict(self.optimizer_config)
        # lr = opt_config.pop("lr", self.learning_rate)
        # opt_name = opt_config.pop("optimizer", "Adam")
        # if not hasattr(torch.optim, opt_name):
        #     raise ValueError(f"Unknown optimizer: {opt_name}")
        # optimizer = getattr(torch.optim, opt_name)(
        #     filter(lambda p: p.requires_grad, self.parameters()), lr=lr, **opt_config
        # )

        # Schedulers
        schedulers = []
        if hasattr(self, "scheduler_config") and self.scheduler_config:
            for sch_cfg in self.scheduler_config:
                sch_cfg = dict(sch_cfg)  # copy
                sch_class = sch_cfg.pop("scheduler")
                # Récupère les paramètres Lightning (et retire-les du dict pour ne pas les passer au scheduler)
                lightning_keys = ["interval", "monitor", "frequency", "strict"]
                lightning_params = {k: sch_cfg.pop(k) for k in lightning_keys if k in sch_cfg}
                # Try torch.optim.lr_scheduler first, then local .optimizers
                # if hasattr(torch.optim.lr_scheduler, sch_name):
                #     sch_class = getattr(torch.optim.lr_scheduler, sch_name)
                # else:
                #     if not hasattr(schedulers, sch_name):
                #         raise ValueError(f"Unknown scheduler: {sch_name}")
                #     sch_class = getattr(schedulers, sch_name)
                scheduler = sch_class(self.opt, **sch_cfg)

                # Lightning scheduler dict
                sch_dict = {"scheduler": scheduler}
                # Ajoute les paramètres Lightning du dict config, sinon valeurs par défaut
                sch_dict["interval"] = lightning_params.get("interval", "epoch")
                if "monitor" in lightning_params:
                    sch_dict["monitor"] = lightning_params["monitor"]
                sch_dict["frequency"] = lightning_params.get("frequency", 1)
                sch_dict["strict"] = lightning_params.get("strict", True)
                schedulers.append(sch_dict)

        if schedulers:
            # returns a list of dict in the "lr_scheduler" key
            # return {"optimizer": optimizer, "lr_scheduler": schedulers}
            return [self.opt], schedulers
        else:
            return self.opt

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["masks"]
        labels = batch["label"]
        category_id = batch["category_id"]
        mass = batch["mass"]
        flat_pred_cls, flat_cls_factor, flat_pred_kernel, seg_preds, geom_features = self.model(images, training=True)
        flat_pred_cls = flat_pred_cls.permute(0, 2, 1)  # [B, ncls, nloc] -> [B, nloc, ncls]

        total_cls_loss, total_seg_loss, total_density_loss = 0, 0, 0
        recalls = []
        precisions = []
        batch_size = images.shape[0]
        for i in range(batch_size):
            # compute targets
            class_targets, label_targets, mass_targets = compute_cls_targets(
                category_id[i],
                masks[i],
                labels[i],
                mass[i],
                images[i].shape[-2:],
                strides=self.strides,
                grid_sizes=self.grid_sizes,
                scale_ranges=self.config.scale_ranges,
                mode="diag",
                offset_factor=self.config.offset_factor,
            )
            # one hot encoding. Delete the bg slice
            class_targets = F.one_hot(class_targets.long(), self.ncls + 1)[..., 1:]

            inputs = (
                class_targets,
                label_targets,
                mass_targets,
                masks[i],
                flat_pred_cls[i],
                flat_cls_factor[i].squeeze(0),
                flat_pred_kernel[i].T,
                seg_preds[i],
                geom_features[i],
            )
            cls_loss, seg_loss, density_loss = compute_image_loss(
                inputs,
                compute_cls_loss=self.losses["cls"],
                compute_seg_loss=self.losses["seg"],
                compute_density_loss=self.losses["mass"],
                seg_loss_func=self.seg_loss,
                max_pos=self.max_pos,
                label_smoothing=self.label_smoothing,
                mask_quality_weighting=self.mask_quality_weighting,
            )
            total_cls_loss += cls_loss
            total_seg_loss += seg_loss
            total_density_loss += density_loss

            # metrics
            recall = Recall(class_targets, flat_pred_cls[i])
            precision = Precision(class_targets, flat_pred_cls[i])
            recalls.append(recall)
            precisions.append(precision)

        # Moyenne sur le batch
        total_cls_loss /= batch_size
        total_seg_loss /= batch_size
        total_density_loss /= batch_size
        total_loss = total_cls_loss + total_seg_loss + total_density_loss
        mean_recall = sum(recalls) / len(recalls)
        mean_precision = sum(precisions) / len(precisions)

        self.log("val_precision", mean_precision, on_epoch=True, prog_bar=True)
        self.log("val_recall", mean_recall, on_epoch=True, prog_bar=True)
        self.log("val_total_loss", total_loss, on_epoch=True, prog_bar=True)
        if self.losses["cls"]:
            self.log("val_cls_loss", total_cls_loss, on_epoch=True, prog_bar=True)
        if self.losses["seg"]:
            self.log("val_seg_loss", total_seg_loss, on_epoch=True, prog_bar=True)
        if self.losses["mass"]:
            self.log("val_density_loss", total_density_loss, on_epoch=True, prog_bar=True)

        return total_loss

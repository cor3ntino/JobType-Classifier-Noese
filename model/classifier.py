import os
import base64

import random

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from torchmetrics import functional as FM

import pytorch_lightning as pl

from transformers import AdamW, get_linear_schedule_with_warmup

from .encoder import Encoder

from .classes import (
    CLASSES_LVL_1, CLASSES_LVL_1_TO_ID, WEIGHTS_CLASSES_LVL_1, POS_WEIGHTS_LVL_1,
    CLASSES_LVL_2, CLASSES_LVL_2_TO_ID, WEIGHTS_CLASSES_LVL_2, POS_WEIGHTS_LVL_2,
    CLASSES_LVL_3, CLASSES_LVL_3_TO_ID, WEIGHTS_CLASSES_LVL_3, POS_WEIGHTS_LVL_3
    
)
from .utils import split_train_val

DEVICE = torch.device("cpu")
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 2e-5

LOG_INTERVAL = 500
TOP_K_CHECKPOINTS = 5

MODEL_NAME = "JobTypeClassifier"
LOGGER_DIR = "model/classifier_logs/"
CHECKPOINTS_DIR = "model/classifier_checkpoints/"

DATASET_DIR             = "/mnt/data/Noese/offres_pole_emploi.json"
DATASET_DIR_PRECOMPUTED = "/mnt/data/Noese/offres_pole_emploi_precomputed.json"

def run_training(lvl, precomputed_embeddings):

    PRECOMPUTED_EMBEDDINGS = precomputed_embeddings

    LVL = lvl

    if LVL == 1:
        CLASSES, CLASSES_TO_ID, WEIGHTS_CLASSES, POS_WEIGHTS = \
            CLASSES_LVL_1, CLASSES_LVL_1_TO_ID, WEIGHTS_CLASSES_LVL_1, POS_WEIGHTS_LVL_1
        # Number of characters used in label 'rome_code' to identify category
        LEN_LABEL_DATASET = 1
    elif LVL == 2:
        CLASSES, CLASSES_TO_ID, WEIGHTS_CLASSES, POS_WEIGHTS = \
            CLASSES_LVL_2, CLASSES_LVL_2_TO_ID, WEIGHTS_CLASSES_LVL_2, POS_WEIGHTS_LVL_2
        LEN_LABEL_DATASET = 3
    elif LVL == 3:
        CLASSES, CLASSES_TO_ID, WEIGHTS_CLASSES, POS_WEIGHTS = \
            CLASSES_LVL_3, CLASSES_LVL_3_TO_ID, WEIGHTS_CLASSES_LVL_3, POS_WEIGHTS_LVL_3
        LEN_LABEL_DATASET = 5
    
    os.makedirs(CHECKPOINTS_DIR, exist_ok = True)

    logger = pl.loggers.TensorBoardLogger(
        LOGGER_DIR,
        name = MODEL_NAME
    )

    data = ClassifierDataModule(
        DATASET_DIR if not PRECOMPUTED_EMBEDDINGS else DATASET_DIR_PRECOMPUTED,
        CLASSES,
        CLASSES_TO_ID,
        LEN_LABEL_DATASET,
        batch_size = BATCH_SIZE,
        precomputed_embeddings=PRECOMPUTED_EMBEDDINGS
    )
    data.setup("fit")
    NUM_TRAINING_DATA = len(data.train_dataloader().dataset)

    wrapper = TrainableCustomClassifier(
        DEVICE,
        LEARNING_RATE,
        NUM_TRAINING_DATA,
        CLASSES,
        WEIGHTS_CLASSES,
        POS_WEIGHTS,
        PRECOMPUTED_EMBEDDINGS
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath = CHECKPOINTS_DIR,
        filename = "{epoch}",
        monitor = "f1_score/validation",
        mode = "max",
        save_top_k = TOP_K_CHECKPOINTS
    )

    trainer = pl.Trainer(
        default_root_dir = CHECKPOINTS_DIR,
        max_epochs = NUM_EPOCHS,
        logger = logger,
        log_every_n_steps = LOG_INTERVAL,
        enable_checkpointing = True,
        callbacks = checkpoint_callback
    )

    trainer.fit(wrapper, data)

    print(f"Best model path : {trainer.checkpoint_callback.best_model_path}")

class CustomClassifier(nn.Module):

    def __init__(self, device, classes, precomputed_embeddings):
        super().__init__()

        self._HIDDEN_DIM = 2 * 384
        self._OUTPUT_DIM = len(classes)

        self.device = device
        self.precomputed_embeddings = precomputed_embeddings
        
        if not precomputed_embeddings:
            self.encoder = Encoder(device=device)
            # Freeze the Transformers part
            for param in self.encoder.model.parameters():
                param.requires_grad = False
            self.encoder.model.eval()

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(self._HIDDEN_DIM, self._HIDDEN_DIM),
            nn.Dropout(0.1),
            nn.Linear(self._HIDDEN_DIM, self._OUTPUT_DIM),
        ).to(device=device)
        for i in (0, 2):
            self.classifier[i].weight.data.normal_(mean = 0.0, std = 0.02)
            self.classifier[i].bias.data.zero_()

    def load_from_local_state_dict(self, state_dict_path):

        state_dict = torch.load(state_dict_path, map_location=self.device)
        self.load_state_dict(state_dict)

    def forward(self, x1, x2):

        if not self.precomputed_embeddings:
            return self.classifier(
                torch.cat(
                    (
                        torch.Tensor(self.encoder.encode(sentences=x1)),
                        torch.Tensor(self.encoder.encode(sentences=x2)),
                    ),
                    dim=1
                )
            )

        else:
            return self.classifier(
                torch.cat((x1, x2), dim=1)
            )

class TrainableCustomClassifier(pl.LightningModule):
    def __init__(
        self,
        device,
        learning_rate,
        num_training_data,
        classes,
        weights_classes,
        pos_weights,
        precomputed_embeddings: bool
    ):  
        super().__init__()

        self.model = CustomClassifier(device, classes, precomputed_embeddings)

        self.weights_classes = weights_classes
        self.pos_weights = pos_weights

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr = self.hparams["learning_rate"],
            weight_decay = 1e-2,
            correct_bias = False
        )

        total_steps = self.hparams["num_training_data"] * NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def _predict(self, intitules, descriptions, targets):
        logits = self(intitules, descriptions)
        loss = F.binary_cross_entropy_with_logits(
            input = logits,
            target = targets,
            weight = self.weights_classes,
            pos_weight = self.pos_weights
        )

        sigmoid = torch.sigmoid(logits)

        return sigmoid, loss

    def training_step(self, batch, batch_idx):
        intitules, descriptions, targets = batch # x, y
        sigmoid, loss = self._predict(intitules, descriptions, targets)
        acc = FM.accuracy(
            preds = sigmoid, target = targets,
            task="multilabel", num_labels=targets.size(1)
        )
        f1_score = FM.f1_score(
            preds = sigmoid, target = targets,
            task="multilabel", num_labels=targets.size(1)
        )

        self.log("loss/train", loss, on_step = True, on_epoch = False)
        self.log("accuracy/train", acc, on_step = True, on_epoch = False)
        self.log("f1_score/train", f1_score, on_step = True, on_epoch = False)

        return loss

    def validation_step(self, batch, batch_idx):
        intitules, descriptions, targets = batch # x, y
        sigmoid, loss = self._predict(intitules, descriptions, targets)
        acc = FM.accuracy(
            preds = sigmoid, target = targets,
            task="multilabel", num_labels=targets.size(1)
        )
        f1_score = FM.f1_score(
            preds = sigmoid, target = targets,
            task="multilabel", num_labels=targets.size(1)
        )

        return {
            "val_loss": loss,
            "accuracy": acc,
            "f1_score": f1_score
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        avg_f1_score = torch.stack([x["f1_score"] for x in outputs]).mean()

        self.log("loss/validation", avg_loss, on_step = False, on_epoch = True)
        self.log("accuracy/validation", avg_acc, on_step = False, on_epoch = True)
        self.log("f1_score/validation", avg_f1_score, on_step = False, on_epoch = True)


class LabeledDataset(Dataset):
    def __init__(
        self,
        data: dict,
        classes,
        classes_to_id,
        len_label_dataset,
        precomputed_embeddings: bool
    ):
        super().__init__()

        self.classes = classes
        self.classes_to_id = classes_to_id
        self.len_label_dataset = len_label_dataset
        
        self._data = list(data.values())
        random.shuffle(self._data)
        self._precomputed_embeddings = precomputed_embeddings

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):

        if not self._precomputed_embeddings:
            intitule = self._data[idx]["intitule"]
            description = self._data[idx]["description"]
        else:
            intitule = torch.from_numpy(
                np.frombuffer(
                    base64.b64decode(self._data[idx]["embedding_intitule_base64"]),
                    dtype=np.float32
                )
            )
            description = torch.from_numpy(
                np.frombuffer(
                    base64.b64decode(self._data[idx]["embedding_description_base64"]),
                    dtype=np.float32
                )
            )

        labels = [
            label[:self.len_label_dataset]
            for label in self._data[idx]["rome_code"]
        ]

        multi_hot_target = torch.zeros(len(self.classes))
        for label in labels:
            multi_hot_target[self.classes_to_id[label]] = 1.

        return (
            intitule,
            description,
            multi_hot_target
        )

    @staticmethod
    def collate(batch):
        intitules = default_collate([i for i, _, _ in batch if i != ""])
        descriptions = default_collate([d for i, d, _ in batch if i != ""])
        targets = default_collate([lbl for i, _, lbl in batch if i != ""])

        return intitules, descriptions, targets

class ClassifierDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        classes,
        classes_to_id,
        len_label_dataset,
        *,
        num_workers: int = 4,
        batch_size: int = 32,
        proportion_validation: float = 0.02,
        precomputed_embeddings: bool = False
    ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.classes = classes
        self.classes_to_id = classes_to_id
        self.len_label_dataset = len_label_dataset

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.proportion_validation = proportion_validation
        self.precomputed_embeddings = precomputed_embeddings

    def setup(self, stage):
        if stage == "fit":
            train_data, val_data = \
                split_train_val(self.dataset_dir, self.proportion_validation)

            self.train_dataset = LabeledDataset(
                train_data,
                self.classes,
                self.classes_to_id,
                self.len_label_dataset,
                self.precomputed_embeddings
            )
            self.val_dataset = LabeledDataset(
                val_data,
                self.classes,
                self.classes_to_id,
                self.len_label_dataset,
                self.precomputed_embeddings
            )

            assert self.train_dataset.classes == self.val_dataset.classes

            self.classes = self.train_dataset.classes

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers = self.num_workers,
            batch_size = self.batch_size,
            shuffle = True,
            collate_fn = LabeledDataset.collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers = self.num_workers,
            batch_size = self.batch_size,
            collate_fn = LabeledDataset.collate
        )


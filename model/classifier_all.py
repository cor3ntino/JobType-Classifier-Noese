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
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

LOG_INTERVAL = 500
TOP_K_CHECKPOINTS = 100

MODEL_NAME = "JobTypeClassifier"
LOGGER_DIR = "model/classifier_logs/"
CHECKPOINTS_DIR = "model/classifier_checkpoints/"

DATASET_DIR             = "/mnt/data/Noese/offres_pole_emploi.json"
DATASET_DIR_PRECOMPUTED = "/mnt/data/Noese/offres_pole_emploi_precomputed.json"

def run_training(precomputed_embeddings):
    """
    Training all 3 levels of ROMEv4's ontology at the same time
    """

    PRECOMPUTED_EMBEDDINGS = precomputed_embeddings

    CLASSES, CLASSES_TO_ID = {}, {}
    WEIGHTS_CLASSES, POS_WEIGHTS = {}, {}
    LEN_LABEL_DATASET= {}
    
    CLASSES["LVL_1"], CLASSES_TO_ID["LVL_1"] = \
        CLASSES_LVL_1, CLASSES_LVL_1_TO_ID
    WEIGHTS_CLASSES["LVL_1"], POS_WEIGHTS["LVL_1"] = \
        WEIGHTS_CLASSES_LVL_1, POS_WEIGHTS_LVL_1
    LEN_LABEL_DATASET["LVL_1"] = 1

    CLASSES["LVL_2"], CLASSES_TO_ID["LVL_2"] = \
        CLASSES_LVL_2, CLASSES_LVL_2_TO_ID
    WEIGHTS_CLASSES["LVL_2"], POS_WEIGHTS["LVL_2"] = \
        WEIGHTS_CLASSES_LVL_2, POS_WEIGHTS_LVL_2
    LEN_LABEL_DATASET["LVL_2"] = 3

    CLASSES["LVL_3"], CLASSES_TO_ID["LVL_3"] = \
        CLASSES_LVL_3, CLASSES_LVL_3_TO_ID
    WEIGHTS_CLASSES["LVL_3"], POS_WEIGHTS["LVL_3"] = \
        WEIGHTS_CLASSES_LVL_3, POS_WEIGHTS_LVL_3
    LEN_LABEL_DATASET["LVL_3"] = 5
    
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
        monitor = "loss_total/validation",
        mode = "min",
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
        self._OUTPUT_DIM = {
            lvl: len(_classes)
            for lvl, _classes
            in classes.items()
        }

        self.device = device
        self.precomputed_embeddings = precomputed_embeddings
        
        if not precomputed_embeddings:
            self.encoder = Encoder(device=device)
            # Freeze the Transformers part
            for param in self.encoder.model.parameters():
                param.requires_grad = False
            self.encoder.model.eval()

        # Classifier layers
        self.classifier_layer_1 = nn.Sequential(
            nn.Linear(self._HIDDEN_DIM, self._HIDDEN_DIM),
            nn.Dropout(0.1),
        ).to(device=device)
        self.classifier_layer_1[0].weight.data.normal_(mean = 0.0, std = 0.02)
        self.classifier_layer_1[0].bias.data.zero_()

        self.classifiers_layer_2 = nn.ModuleDict({
            lvl: nn.Linear(self._HIDDEN_DIM, output_dim).to(device=device)
            for lvl, output_dim
            in self._OUTPUT_DIM.items()
        })
        for _, linear_layer_2 in self.classifiers_layer_2.items():
            linear_layer_2.weight.data.normal_(mean = 0.0, std = 0.02)
            linear_layer_2.bias.data.zero_()
        
    def load_from_local_state_dict(self, state_dict_path):

        state_dict = torch.load(state_dict_path, map_location=self.device)
        self.load_state_dict(state_dict)

    def forward(self, x1, x2):

        if not self.precomputed_embeddings:

            concat_embs = torch.cat(
                (
                    torch.Tensor(self.encoder.encode(sentences=x1)),
                    torch.Tensor(self.encoder.encode(sentences=x2)),
                ),
                dim=1
            )

            output_layer_1 = self.classifier_layer_1(concat_embs)

            outputs_layer_2 = {
                lvl: linear_layer_2(output_layer_1)
                for lvl, linear_layer_2 in self.classifiers_layer_2.items()
            }

            return outputs_layer_2

        else:

            concat_embs = torch.cat((x1, x2), dim=1)

            output_layer_1 = self.classifier_layer_1(concat_embs)

            outputs_layer_2 = {
                lvl: linear_layer_2(output_layer_1)
                for lvl, linear_layer_2 in self.classifiers_layer_2.items()
            }

            return outputs_layer_2

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
        loss = {
            lvl: F.binary_cross_entropy_with_logits(
                input = _logits,
                target = targets[lvl],
                weight = self.weights_classes[lvl],
                pos_weight = self.pos_weights[lvl]
            )
            for lvl, _logits in logits.items()
        }

        sigmoid = {
            lvl: torch.sigmoid(_logits)
            for lvl, _logits in logits.items()
        }

        return sigmoid, loss

    def training_step(self, batch, batch_idx):
        intitules, descriptions, targets = batch # x, y
        sigmoid, loss = self._predict(intitules, descriptions, targets)

        loss_total = 0.
        for lvl, _loss in loss.items():
            self.log(f"loss_{lvl}/train", _loss, on_step = True, on_epoch = False)
            loss_total += _loss
        self.log("loss_total/train", loss_total, on_step = True, on_epoch = False)

        for lvl, _sigmoid in sigmoid.items():
            acc = FM.accuracy(
                preds = _sigmoid, target = targets[lvl],
                task="multilabel", num_labels=targets[lvl].size(1)
            )
            f1_score = FM.f1_score(
                preds = _sigmoid, target = targets[lvl],
                task="multilabel", num_labels=targets[lvl].size(1)
            )

            self.log(f"accuracy_{lvl}/train", acc, on_step = True, on_epoch = False)
            self.log(f"f1_score_{lvl}/train", f1_score, on_step = True, on_epoch = False)

        return loss_total

    def validation_step(self, batch, batch_idx):
        intitules, descriptions, targets = batch # x, y
        sigmoid, loss = self._predict(intitules, descriptions, targets)

        res = {}

        for lvl, _sigmoid in sigmoid.items():
            acc = FM.accuracy(
                preds = _sigmoid, target = targets[lvl],
                task="multilabel", num_labels=targets[lvl].size(1)
            )
            res[f"accuracy_{lvl}"] = acc
            f1_score = FM.f1_score(
                preds = _sigmoid, target = targets[lvl],
                task="multilabel", num_labels=targets[lvl].size(1)
            )
            res[f"f1_score_{lvl}"] = f1_score

        loss_total = 0.
        for lvl, _loss in loss.items():
            res[f"loss_{lvl}"] = _loss
            loss_total += _loss
        res["loss_total"] = loss_total
        
        return res

    def validation_epoch_end(self, outputs):

        for key in outputs[0].keys():
            avg = torch.stack([x[key] for x in outputs]).mean()
            self.log(f"{key}/validation", avg, on_step = False, on_epoch = True)

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

        labels = self._data[idx]["rome_code"]

        multi_hot_target = {
            lvl: torch.zeros(len(_classes))
            for lvl, _classes in self.classes.items()
        }
        for label in labels:
            for lvl, _classes_to_id in self.classes_to_id.items():
                multi_hot_target[lvl][
                    _classes_to_id[
                        label[:self.len_label_dataset[lvl]]
                    ]
                ] = 1.

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


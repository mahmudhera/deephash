import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import os

# Parameters
K = 31
INPUT_DIM = 4 * K
VARIANTS_PER_KMER = 10  # Number of variants per base k-mer
GROUP_SIZE = VARIANTS_PER_KMER + 1
BATCH_SIZE = 256
MAX_EPOCHS = 10
LEARNING_RATE = 1e-3

# Dataset
class KmerDataset(Dataset):
    def __init__(self, csv_path, use_targets=True):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df.drop(columns=['target']).values, dtype=torch.float32)
        self.y = torch.tensor(df['target'].values, dtype=torch.float32) if use_targets else None
        self.use_targets = use_targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.use_targets:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# LSH model
class LSHModel(LightningModule):
    def __init__(self, lr=LEARNING_RATE):
        super().__init__()
        self.save_hyperparameters()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(INPUT_DIM, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()  # ensures hash output is in [0, 1]
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        return y_hat.detach()

    def on_validation_epoch_start(self):
        self.validation_preds = []

    def on_validation_epoch_end(self):
        preds = torch.cat(self.validation_preds, dim=0).numpy()
        groups = preds.reshape(-1, GROUP_SIZE)

        # Intra-group variance
        intra_var = np.mean(np.var(groups, axis=1))

        # Histogram stddev of base hashes
        base_vals = groups[:, 0]
        hist, _ = np.histogram(base_vals, bins=10, range=(0, 1))
        uniformity = np.std(hist / np.sum(hist))

        self.log("val_intra_group_var", intra_var)
        self.log("val_base_uniformity", uniformity)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def main():
    train_ds = KmerDataset("train.csv", use_targets=True)
    val_ds = KmerDataset("valid.csv", use_targets=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSHModel()

    checkpoint_cb = ModelCheckpoint(
        monitor="val_intra_group_var",
        save_top_k=1,
        mode="min",
        filename="best-lsh-model-{epoch:02d}-{val_intra_group_var:.5f}"
    )

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_cb],
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()

from pathlib import Path
import math


import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from aaindex_encoding import AAINDEX_DIM
from dataloader import MissenseCollator, MissenseDataset
from model import CNNPairClassifier, GatedResidualClassifier, MLPClassifier, MissenseESMBiLSTM


ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = ROOT / "data/proceed/splits/train.csv"
VAL_CSV = ROOT / "data/proceed/splits/val.csv"

# Experiment tag: used to name the saved model and TensorBoard log sub-directory.
TAG = "exp9_final"
CLASSIFIER_TYPE = "mlp" # "mlp", "cnn", or "gated"

SAVE_DIR = ROOT / "model"
SAVE_NAME = f"{CLASSIFIER_TYPE}_{TAG}.pt"
LOG_DIR = ROOT / "result" / "tensorboard" / TAG

PLM_NAME = "facebook/esm2_t33_650M_UR50D"
EPOCHS = 15
BATCH_SIZE = 32
LR = 6.481308012346065e-5
NUM_WORKERS = 8
FREEZE_ESM = True
LOCAL_FILES_ONLY = True

LSTM_HIDDEN = 256
PROJ_DIM = 64
DROPOUT = 0.5382680063323042
SEED = 42
USE_AMP = True
WEIGHT_DECAY = 0.005135321670338704
THRESHOLD = 0.5

# 3 type of different classifiers to compare:
def build_classifier(input_dim, proj_dim, dropout):
    hidden_dim = max(proj_dim // 2, 64)
    if CLASSIFIER_TYPE == "mlp":
        return MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if CLASSIFIER_TYPE == "cnn":
        return CNNPairClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if CLASSIFIER_TYPE == "gated":
        return GatedResidualClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unsupported CLASSIFIER_TYPE: {CLASSIFIER_TYPE}")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if hasattr(value, "to") else value
    return out


def get_loader(csv_path, shuffle):
    dataset = MissenseDataset(csv_path)
    collator = MissenseCollator(PLM_NAME, local_files_only=LOCAL_FILES_ONLY)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

# warmup learning rate scheduler
def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch+1) / float(max(1, warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# metrics 
def calc_metrics(labels, probs):
    # set threshold at 0.5 for binary classification
    preds = (probs >= THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "auroc": roc_auc_score(labels, probs),
        "auprc": average_precision_score(labels, probs),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
            logits = model(batch)
            loss = criterion(logits, batch["labels"])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch["labels"].size(0)

    return total_loss / len(loader.dataset)

# Evaluation without gradient calculation for efficiency
@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="val"):
    model.eval()
    total_loss = 0.0
    labels = []
    probs = []

    for batch in tqdm(loader, desc=desc, leave=False):
        batch = move_batch(batch, device)
        logits = model(batch)
        loss = criterion(logits, batch["labels"])

        total_loss += loss.item() * batch["labels"].size(0)
        labels.append(batch["labels"])
        probs.append(torch.sigmoid(logits))

    labels = torch.cat(labels).cpu().numpy()
    probs = torch.cat(probs).cpu().numpy()
    metrics = calc_metrics(labels, probs)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics




def create_training_objects(lstm_hidden, proj_dim, dropout, lr, weight_decay, device):
    classifier = build_classifier(proj_dim * 4, proj_dim, dropout)
    model = MissenseESMBiLSTM(
        plm_name=PLM_NAME,
        aaindex_dim=AAINDEX_DIM,
        lstm_hidden=lstm_hidden,
        proj_dim=proj_dim,
        dropout=dropout,
        freeze_esm=FREEZE_ESM,
        local_files_only=LOCAL_FILES_ONLY,
        classifier=classifier,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=2, total_epochs=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda")

    return model, criterion, optimizer, scheduler, scaler

# use tensorboard to log metric
def tensorboard_log(writer, epoch, metrics, split):
    for key, value in metrics.items():
        writer.add_scalar(f"{split}/{key}", value, epoch)


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, val_metrics):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "val_metrics": val_metrics,
            "config": {
                "tag": TAG,
                "classifier_type": CLASSIFIER_TYPE,
                "plm_name": PLM_NAME,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "lstm_hidden": LSTM_HIDDEN,
                "proj_dim": PROJ_DIM,
                "dropout": DROPOUT,
                "freeze_esm": FREEZE_ESM,
                "threshold": THRESHOLD,
            },
        },
        SAVE_DIR / SAVE_NAME,
    )


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    train_loader = get_loader(TRAIN_CSV, shuffle=True)
    val_loader = get_loader(VAL_CSV, shuffle=False)

    model, criterion, optimizer, scheduler, scaler = create_training_objects(
        lstm_hidden=LSTM_HIDDEN,
        proj_dim=PROJ_DIM,
        dropout=DROPOUT,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device=device,
    )

    writer = SummaryWriter(LOG_DIR)
    best_loss = float("inf")

    try:
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
            val_metrics = evaluate(model, val_loader, criterion, device, desc="val")

            tensorboard_log(writer, epoch, val_metrics, "val")
            writer.add_scalar("train/optimization_loss", train_loss, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                save_checkpoint(model, optimizer, scheduler, epoch, best_loss, val_metrics)

            scheduler.step()

            print(
                f"Epoch {epoch}/{EPOCHS} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_auroc={val_metrics['auroc']:.4f} "
                f"val_auprc={val_metrics['auprc']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} "
                f"val_mcc={val_metrics['mcc']:.4f}"
            )
    finally:
        writer.close()

    print(f"Best val loss: {best_loss:.4f}")
    print(f"Saved best checkpoint to: {SAVE_DIR / SAVE_NAME}")

if __name__ == "__main__":
    main()

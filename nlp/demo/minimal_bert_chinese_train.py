"""
Minimal local training script for Chinese product title classification.

Run from the repository root:
    python minimal_bert_chinese_train.py

This script is intentionally small and explicit. It reads the raw TSV files,
builds label mappings, tokenizes text with bert-base-chinese, trains for a few
steps, evaluates once per epoch, and saves the tiny demo model.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


ROOT_DIR = Path(__file__).parent
TRAIN_FILE = ROOT_DIR / "data" / "raw" / "train.txt"
VALID_FILE = ROOT_DIR / "data" / "raw" / "valid.txt"
OUTPUT_DIR = ROOT_DIR / "checkpoint" / "minimal-demo"

MODEL_NAME = "google-bert/bert-base-chinese"
MAX_LENGTH = 64
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5

# Keep these small on purpose. Increase them after you understand the flow.
MAX_TRAIN_SAMPLES = 256
MAX_VALID_SAMPLES = 64

# True means only train the classification head. This is faster for learning
# the training loop. Set False when you want real fine-tuning.
FREEZE_BERT = True


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_tsv(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({"label": row["label"], "text": row["text_a"]})
            if limit is not None and len(rows) >= limit:
                break
    return rows


class ProductTitleDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        tokenizer: AutoTokenizer,
        label2id: dict[str, int],
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = self.label2id[row["label"]]
        return encoded


def evaluate(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)

            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    model.train()
    return avg_loss, accuracy


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    train_rows = read_tsv(TRAIN_FILE, MAX_TRAIN_SAMPLES)
    valid_rows = read_tsv(VALID_FILE, MAX_VALID_SAMPLES)

    labels = sorted({row["label"] for row in train_rows})
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    print(f"Train samples: {len(train_rows)}")
    print(f"Valid samples: {len(valid_rows)}")
    print(f"Labels in tiny run: {len(labels)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    if FREEZE_BERT:
        for param in model.bert.parameters():
            param.requires_grad = False
        print("Frozen BERT encoder: only the classification head will train.")

    model.to(device)

    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_loader = DataLoader(
        ProductTitleDataset(train_rows, tokenizer, label2id),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        ProductTitleDataset(valid_rows, tokenizer, label2id),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 10 == 0:
                print(f"epoch={epoch + 1} step={global_step} train_loss={loss.item():.4f}")

        valid_loss, valid_acc = evaluate(model, valid_loader, device)
        print(
            f"epoch={epoch + 1} "
            f"valid_loss={valid_loss:.4f} "
            f"valid_acc={valid_acc:.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved demo model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
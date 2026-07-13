"""BERT 中文商品分类：180 行核心训练脚本填空大纲。

目标：你根据这个骨架，把每个 TODO 补完整，最终写出一个可运行的最小训练脚本。
运行位置：项目根目录
参考答案：minimal_bert_chinese_train.py
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

# =============================================================================
# 1. 全局配置：路径、模型名、训练超参数
# =============================================================================

ROOT_DIR = Path(__file__).parent
TRAIN_FILE = ROOT_DIR / "data" / "raw" / "train.txt"
VALID_FILE = ROOT_DIR / "data" / "raw" / "valid.txt"
OUTPUT_DIR = ROOT_DIR / "checkpoint" / "minimal-demo"

MODEL_NAME = "google-bert/bert-base-chinese"
MAX_LENGTH = 64
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5
MAX_TRAIN_SAMPLES = 256
MAX_VALID_SAMPLES = 64
FREEZE_BERT = True


# =============================================================================
# 2. 选择设备：Mac MPS / CUDA / CPU
# =============================================================================

def get_device() -> torch.device:
  """返回当前机器可用的最佳训练设备。"""
  # TODO:
  # 1. 如果 torch.backends.mps.is_available()，返回 torch.device("mps")
  # 2. 否则如果 torch.cuda.is_available()，返回 torch.device("cuda")
  # 3. 否则返回 torch.device("cpu")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")
  # raise NotImplementedError


# =============================================================================
# 3. 读取 TSV 原始数据
# =============================================================================

def read_tsv(path: Path, limit: int | None = None) -> list[dict[str, str]]:
  """读取 train.txt / valid.txt，返回 [{"label": "...", "text": "..."}]。"""
  rows: list[dict[str, str]] = []

  # TODO:
  # 1. 用 path.open("r", encoding="utf-8", newline="") 打开文件
  # 2. 用 csv.DictReader(f, delimiter="\t") 读取 TSV
  # 3. 每一行取 row["label"] 和 row["text_a"]
  # 4. 追加成 {"label": ..., "text": ...}
  # 5. 如果达到 limit，就 break

  with path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
      rows.append({"label": row["label"], "text": row["text_a"]})
      if limit is not None and len(rows) >= limit:
        break

  return rows


# =============================================================================
# 4. 自定义 Dataset：一条样本 -> BERT 输入
# =============================================================================

class ProductTitleDataset(Dataset):
  """把原始样本转换成 input_ids / attention_mask / labels。"""

  def __init__(self, rows, tokenizer, label2id) -> None:
    # TODO:
    # 保存 rows / tokenizer / label2id 到 self
    self.rows = rows
    self.tokenizer = tokenizer
    self.label2id = label2id
    self.id2label = {index: label for label, index in label2id.items()}
    # raise NotImplementedError

  def __len__(self) -> int:
    # TODO:
    # 返回样本数量
    return len(self.rows)
    # raise NotImplementedError

  def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
    # TODO:
    # 1. 取出第 index 条 row
    # 2. tokenizer(row["text"], truncation=True, max_length=MAX_LENGTH)
    # 3. encoded["labels"] = self.label2id[row["label"]]
    # 4. 返回 encoded

    row = self.rows[index]
    encoded = self.tokenizer(
      row["text"],
      truncation=True,
      max_length=MAX_LENGTH,
    )
    encoded["labels"] = self.label2id[row["label"]]
    return encoded

    # raise NotImplementedError


# =============================================================================
# 5. 验证函数：只评估，不更新参数
# =============================================================================

def evaluate(model, dataloader, device) -> tuple[float, float]:
  """返回验证集平均 loss 和 accuracy。"""
  # TODO:
  # 1. model.eval()
  # 2. 初始化 total_loss / correct / total
  # 3. with torch.no_grad():
  # 4. 遍历 dataloader
  # 5. batch 移到 device
  # 6. outputs = model(**batch)
  # 7. 累加 outputs.loss.item()
  # 8. preds = outputs.logits.argmax(dim=-1)
  # 9. 统计 correct 和 total
  # 10. model.train()
  # 11. 返回 avg_loss, accuracy
  
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
    
  # raise NotImplementedError


# =============================================================================
# 6. 主流程：读数据 -> 建模型 -> 建 DataLoader -> 训练 -> 验证 -> 保存
# =============================================================================

def main() -> None:
  # -------------------------------------------------------------------------
  # 6.1 选择设备
  # -------------------------------------------------------------------------
  # TODO:
  device = get_device()
  print(f"device: {device}")

  # -------------------------------------------------------------------------
  # 6.2 读取少量训练集和验证集
  # -------------------------------------------------------------------------
  # TODO:
  train_rows = read_tsv(TRAIN_FILE, MAX_TRAIN_SAMPLES)
  valid_rows = read_tsv(VALID_FILE, MAX_VALID_SAMPLES)

  print(f"train samples: {len(train_rows)}")
  print(f"valid samples: {len(valid_rows)}")

  # -------------------------------------------------------------------------
  # 6.3 构造标签映射
  # -------------------------------------------------------------------------
  # TODO:
  # labels = sorted(...)
  # label2id = ...
  # id2label = ...
  # print 样本数、类别数
  labels = sorted({row["label"] for row in train_rows})
  label2id = {label: idx for idx, label in enumerate(labels)}
  id2label = {idx: label for idx, label in enumerate(labels)}
  print(f"samples: {len(train_rows)}")

  # -------------------------------------------------------------------------
  # 6.4 加载 tokenizer 和 BERT 分类模型
  # -------------------------------------------------------------------------
  # TODO:
  # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  # model = AutoModelForSequenceClassification.from_pretrained(...)

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

  # -------------------------------------------------------------------------
  # 6.5 可选：冻结 BERT 主体，只训练分类头
  # -------------------------------------------------------------------------
  # TODO:
  # if FREEZE_BERT:
  #     for param in model.bert.parameters():
  #         param.requires_grad = False
  # model.to(device)

  if FREEZE_BERT:
    for param in model.bert.parameters():
      param.requires_grad = False
    print("Frozen BERT encoder: only the classification head will train.")
  model.to(device)

  # -------------------------------------------------------------------------
  # 6.6 创建 Dataset / DataLoader / collate_fn
  # -------------------------------------------------------------------------
  # TODO:
  # collate_fn = DataCollatorWithPadding(...)
  # train_loader = DataLoader(...)
  # valid_loader = DataLoader(...)

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

  # -------------------------------------------------------------------------
  # 6.7 创建优化器
  # -------------------------------------------------------------------------
  # TODO:
  # optimizer = torch.optim.AdamW(
  #     filter(lambda param: param.requires_grad, model.parameters()),
  #     lr=LEARNING_RATE,
  # )

  optimizer = torch.optim.AdamW(
    filter(lambda param: param.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
  )

  # -------------------------------------------------------------------------
  # 6.8 训练循环：epoch -> batch -> loss -> backward -> step
  # -------------------------------------------------------------------------
  # TODO:
  # global_step = 0
  # for epoch in range(EPOCHS):
  #     model.train()
  #     for batch in train_loader:
  #         batch = ...
  #         outputs = model(**batch)
  #         loss = outputs.loss
  #         loss.backward()
  #         optimizer.step()
  #         optimizer.zero_grad()
  #         global_step += 1
  #         每 10 step 打印一次 loss

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
      
  # -------------------------------------------------------------------------
  # 6.9 每个 epoch 结束后验证
  # -------------------------------------------------------------------------
  # TODO:
  # valid_loss, valid_acc = evaluate(model, valid_loader, device)
  # print(...)
  
  valid_loss, valid_acc = evaluate(model, valid_loader, device)
  
  print(
    f"epoch={epoch + 1} "
    f"valid_loss={valid_loss:.4f} "
    f"valid_acc={valid_acc:.4f}"
  )

  # -------------------------------------------------------------------------
  # 6.10 保存模型和 tokenizer
  # -------------------------------------------------------------------------
  # TODO:
  # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  # model.save_pretrained(OUTPUT_DIR)
  # tokenizer.save_pretrained(OUTPUT_DIR)
  
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  model.save_pretrained(OUTPUT_DIR)
  tokenizer.save_pretrained(OUTPUT_DIR)
  print(f"Saved demo model to: {OUTPUT_DIR}")
  
  return
  # raise NotImplementedError


if __name__ == "__main__":
  main()

"""模型训练脚本

流程：加载数据 → 构建模型 → 按 epoch 迭代训练 → 保存最优模型权重

运行方式：
    python train.py
"""

import time
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import MODEL_DIR, VOCAB_FILE, BEST_MODEL_FILE, LOG_DIR, LEARNING_RATE, EPOCHS, DEVICE
from dataset import get_dataloader
from model import InputMethodModel
from tokenizer import JiebaTokenizer


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> float:
    """训练一个 epoch，返回该 epoch 的平均损失。"""
    model.train()   # 启用训练模式（开启 Dropout 等正则化层）
    total_loss = 0.0

    for input_ids, target_ids in tqdm(dataloader, desc='训练'):
        input_ids  = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)               # 前向传播
        loss   = criterion(logits, target_ids)  # 计算交叉熵损失

        # 标准三步：清零梯度 → 反向传播 → 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = DEVICE
    print(f"使用设备：{device}")

    train_loader = get_dataloader('train')

    # 词表大小决定了 Embedding 层和输出层的维度
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR / VOCAB_FILE)
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)

    # CrossEntropyLoss 内部已包含 softmax，模型输出原始 logits 即可
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 每次训练用时间戳区分不同实验的 TensorBoard 日志
    writer = SummaryWriter(log_dir=LOG_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    best_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*10} Epoch {epoch}/{EPOCHS} {'='*10}")
        avg_loss =   (model, train_loader, criterion, optimizer, device)
        print(f"平均训练损失：{avg_loss:.4f}")

        writer.add_scalar('Loss/train', avg_loss, epoch)

        # 损失下降则保存当前权重（Model Checkpoint 策略）
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_DIR / BEST_MODEL_FILE)
            print("  → 已保存最优模型")

    writer.close()
    print(f"\n训练完成！最优损失：{best_loss:.4f}")


if __name__ == '__main__':
    main()

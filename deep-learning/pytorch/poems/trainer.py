"""
训练模块
封装单轮训练和完整训练流程
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def train_one_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm):
    """
    执行一轮（epoch）训练，返回该轮的样本级平均损失。

    每个 mini-batch 的训练步骤：
      1. 数据移至目标设备（GPU/CPU）
      2. 前向传播 → 得到 logits
      3. 计算交叉熵损失
      4. 反向传播 → 累积梯度
      5. 梯度裁剪 → 防止梯度爆炸
      6. 优化器更新参数
      7. 梯度清零 → 为下一个 batch 准备

    Args:
        model:         待训练的模型（PoemRNNLM）
        dataloader:    训练数据加载器
        optimizer:     优化器（如 Adam）
        criterion:     损失函数（如 CrossEntropyLoss）
        device:        计算设备
        max_grad_norm: 梯度裁剪的 L2 范数上限

    Returns:
        avg_loss: 该 epoch 的样本级平均损失
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)   # x, y 形状均为 (N, L)

        # 前向传播：logits 形状为 (N, L, vocab_size)
        logits, _ = model(x)

        # CrossEntropyLoss 要求：预测形状 (N, C, L)，目标形状 (N, L)
        # 通过 transpose(1, 2) 将 (N, L, C) → (N, C, L)
        loss = criterion(logits.transpose(1, 2), y)

        # 反向传播：计算各参数的梯度
        loss.backward()

        # 梯度裁剪：将所有参数梯度的全局 L2 范数截断到 max_grad_norm
        # RNN 训练中梯度可能沿时间步指数增长（梯度爆炸），此操作是常见的稳定手段
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        # 用样本数加权累积损失，以便最终求样本级平均（而非 batch 平均）
        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # ASCII 进度条
        progress = int((i + 1) / len(dataloader) * 50)
        print(f"\r[{'=' * progress:<50}]", end='')

    return total_loss / total_samples


def train(model, dataset, config, device):
    """
    完整训练流程。

    Args:
        model:   PoemRNNLM 模型实例
        dataset: PoemDataset 数据集
        config:  配置模块（需含 BATCH_SIZE, LEARNING_RATE, EPOCH_NUM, MAX_GRAD_NORM）
        device:  计算设备

    Returns:
        model: 训练完成的模型
    """
    # shuffle=True：每轮打乱顺序，避免模型记住 batch 的排列规律
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Adam：自适应学习率优化器，对 NLP 任务通常比 SGD 收敛更快更稳
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 交叉熵损失：等价于负对数似然（NLL），是语言模型的标准训练目标
    criterion = nn.CrossEntropyLoss()

    print(f"开始训练：{config.EPOCH_NUM} 轮 | "
          f"批大小 {config.BATCH_SIZE} | "
          f"学习率 {config.LEARNING_RATE}\n")

    for epoch in range(config.EPOCH_NUM):
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, criterion, device, config.MAX_GRAD_NORM
        )
        print(f" epoch {epoch + 1:>2}/{config.EPOCH_NUM}  loss: {avg_loss:.6f}")

    return model

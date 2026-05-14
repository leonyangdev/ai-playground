"""
主程序入口
整合数据加载、模型构建、训练和诗歌生成的完整流程。

运行方式：
    python main.py
"""
import torch

import config
from dataset import load_data
from model import PoemRNNLM
from trainer import train
from generator import generate_poem


def main():
    # ── 1. 数据加载 ──────────────────────────────────────────────────────────
    print("正在加载数据...")
    dataset, id2word, word2id = load_data(config.DATA_PATH, config.SEQ_LEN)
    vocab_size = len(id2word)
    print(f"词表大小: {vocab_size} | 训练样本数: {len(dataset)}\n")

    # ── 2. 设备配置 ──────────────────────────────────────────────────────────
    # 优先级：CUDA（NVIDIA GPU）→ MPS（Apple Silicon）→ CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}\n")

    # ── 3. 模型构建 ──────────────────────────────────────────────────────────
    model = PoemRNNLM(
        vocab_size=vocab_size,
        embedding_size=config.EMBEDDING_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}\n")

    # ── 4. 训练 ──────────────────────────────────────────────────────────────
    train(model, dataset, config, device)

    # ── 5. 生成古诗 ──────────────────────────────────────────────────────────
    print("\n── 生成示例（七言绝句）──")
    for start_token in ["春", "山", "月", "风", "花"]:
        poem = generate_poem(
            model, id2word, word2id,
            start_token=start_token,
            line_num=config.DEFAULT_LINE_NUM,
            line_len=config.DEFAULT_LINE_LEN,
            temperature=config.DEFAULT_TEMPERATURE,
            device=device
        )
        print(f"\n【{start_token}】\n{poem}")


if __name__ == "__main__":
    main()

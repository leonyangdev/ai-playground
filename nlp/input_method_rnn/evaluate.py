"""模型评估脚本

在测试集上计算 Top-1 和 Top-5 准确率：
    Top-1 acc：预测概率最高的词恰好是目标词的比例
    Top-5 acc：目标词出现在前 5 个候选词中的比例

输入法场景下 Top-5 更有实用意义（候选词列表通常显示 5 个）。

运行方式：
    python evaluate.py
"""

import torch
from tqdm import tqdm

from config import MODEL_DIR, VOCAB_FILE, BEST_MODEL_FILE, DEVICE
from dataset import get_dataloader
from model import InputMethodModel
from tokenizer import JiebaTokenizer


def evaluate(model, dataloader, device, k: int = 5) -> tuple[float, float]:
    """在给定数据集上计算 Top-1 和 Top-5 准确率。"""
    model.eval()
    top1_correct = top5_correct = total = 0

    with torch.no_grad():
        for input_ids, target_ids in tqdm(dataloader, desc='评估'):
            input_ids  = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits   = model(input_ids)                     # (N, vocab_size)
            top_k    = torch.topk(logits, k).indices        # (N, k)

            for target, top_k_ids in zip(target_ids.tolist(), top_k.tolist()):
                total += 1
                if target == top_k_ids[0]:
                    top1_correct += 1
                if target in top_k_ids:
                    top5_correct += 1

    return top1_correct / total, top5_correct / total


def main():
    device = DEVICE

    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR / VOCAB_FILE)
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / BEST_MODEL_FILE, map_location=device))

    test_loader = get_dataloader('test')
    top1_acc, top5_acc = evaluate(model, test_loader, device)

    print(f"\n评估结果（测试集）：")
    print(f"  Top-1 准确率：{top1_acc:.4f}  ({top1_acc * 100:.2f}%)")
    print(f"  Top-5 准确率：{top5_acc:.4f}  ({top5_acc * 100:.2f}%)")


if __name__ == '__main__':
    main()

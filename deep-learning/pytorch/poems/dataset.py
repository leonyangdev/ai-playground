"""
数据预处理模块
职责：
  1. 读取原始语料，构建字级别词表
  2. 将文本 ID 化（字 → 整数索引）
  3. 提供 PyTorch Dataset 接口，生成 (x, y) 训练样本对
"""
import re
import torch
from torch.utils.data import Dataset

UNK_TOKEN = "<UNK>"   # 未登录词标记（Out-of-Vocabulary）


def build_vocab(file_path):
    """
    读取古诗语料，构建字级别词表。

    处理流程：
      - 逐行读取 → 去标点 → 收集字符集 → 构建双向映射

    Args:
        file_path: 语料库路径，每行一首诗

    Returns:
        poems:   list[list[str]]，每首诗拆分为字的列表
        id2word: list[str]，下标即 ID，值为对应的字
        word2id: dict[str, int]，字到 ID 的映射
    """
    poems = []
    char_set = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去除标点（保留汉字），并去掉首尾空白
            line = re.sub(r"[，。、？！：]", "", line).strip()
            if not line:
                continue
            char_set.update(list(line))
            poems.append(list(line))

    # 排序保证每次运行词表顺序一致，末尾追加特殊标记
    id2word = sorted(list(char_set)) + [UNK_TOKEN]
    word2id = {word: idx for idx, word in enumerate(id2word)}

    return poems, id2word, word2id


def encode_poems(poems, word2id):
    """
    将每首诗的字符序列转换为 ID 序列（语料 ID 化）。

    Args:
        poems:   build_vocab 返回的字符列表
        word2id: 字到 ID 的映射字典

    Returns:
        poems_id: list[list[int]]，每首诗对应的 ID 序列
    """
    unk_id = word2id[UNK_TOKEN]
    return [
        [word2id.get(char, unk_id) for char in poem]
        for poem in poems
    ]


class PoemDataset(Dataset):
    """
    古诗语言模型数据集。

    用滑动窗口将连续 ID 序列切分为 (x, y) 样本对：
        x = poem_id[i : i+seq_len]      ← 输入序列
        y = poem_id[i+1 : i+1+seq_len]  ← 目标序列（x 整体右移一位）

    语言模型训练目标：给定 x，预测 y（即下一个字），
    本质上是最大化 P(y_t | x_1, ..., x_t) 的联合概率。
    """

    def __init__(self, poems_id, seq_len):
        """
        Args:
            poems_id: encode_poems 返回的 ID 序列列表
            seq_len:  每个训练样本的序列长度 L
        """
        self.samples = []   # 存放 (x_ids, y_ids) 元组

        for poem_id in poems_id:
            # 一首诗至少需要 seq_len+1 个字才能切出一个有效样本
            for i in range(len(poem_id) - seq_len):
                x = poem_id[i:     i + seq_len]
                y = poem_id[i + 1: i + 1 + seq_len]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # LongTensor 用于 Embedding 层的整数索引
        return torch.LongTensor(x), torch.LongTensor(y)


def load_data(file_path, seq_len):
    """
    一步完成数据加载的便捷函数。

    Returns:
        dataset: PoemDataset 实例
        id2word: ID → 字 的映射列表
        word2id: 字 → ID 的映射字典
    """
    poems, id2word, word2id = build_vocab(file_path)
    poems_id = encode_poems(poems, word2id)
    dataset = PoemDataset(poems_id, seq_len)
    return dataset, id2word, word2id

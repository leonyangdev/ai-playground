"""
模型定义模块
使用 RNN（循环神经网络）构建字级语言模型（Character-level Language Model）

网络架构：
    输入 (N, L) → Embedding → RNN（多层） → Dropout → Linear → 输出 (N, L, V)

    N = batch size，L = 序列长度，V = 词表大小
"""
import torch.nn as nn


class PoemRNNLM(nn.Module):
    """
    基于多层 RNN 的古诗语言模型。

    各层说明：
      - Embedding：将离散字 ID 映射为稠密连续向量（可学习的词嵌入表）
      - RNN：对字序列建模时序依赖，隐藏状态 h_t 编码了 t 时刻之前的全部上下文
      - Dropout：训练时随机置零神经元，缓解过拟合
      - Linear：将 RNN 隐藏状态投影到词表空间，输出每个位置的 logits（未归一化分数）
    """

    def __init__(self, vocab_size, embedding_size=128, hidden_size=256,
                 num_layers=1, dropout=0.0):
        """
        Args:
            vocab_size:     词表大小（决定 Embedding 输入维和 Linear 输出维）
            embedding_size: 词嵌入维度
            hidden_size:    RNN 每层的隐藏单元数
            num_layers:     RNN 堆叠层数（≥2 时层间自动加 Dropout）
            dropout:        Dropout 概率（0.0 表示不使用）
        """
        super().__init__()

        # 词嵌入层：vocab_size 个字，每个字用 embedding_size 维向量表示
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size
        )

        # 多层 RNN：batch_first=True 使张量形状为 (N, L, H)，更直观
        # RNN 层间 Dropout 只在 num_layers > 1 时生效
        self.rnn = nn.RNN(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 输出前的 Dropout，在全连接层之前随机丢弃隐藏状态中的信息
        self.dropout = nn.Dropout(dropout)

        # 输出层：将隐藏向量映射到词表维度，得到每个字的 logits
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size
        )

    def forward(self, x, hx=None):
        """
        前向传播。

        Args:
            x:  输入 token ID 序列，形状 (N, L)
            hx: 初始隐藏状态，形状 (num_layers, N, H)；
                None 表示全零初始化（适用于每个新样本的第一步）

        Returns:
            logits: 每个位置的词表 logits，形状 (N, L, vocab_size)
            hidden: RNN 最终隐藏状态，形状 (num_layers, N, H)
                    推理时需将此值传入下一步，以保留生成上下文
        """
        # (N, L) → (N, L, embedding_size)：查嵌入表，将 ID 转为向量
        embedded = self.embedding(x)

        # (N, L, embedding_size) → output: (N, L, H), hidden: (num_layers, N, H)
        output, hidden = self.rnn(embedded, hx)

        # Dropout 正则化：仅在训练阶段生效，eval() 后自动跳过
        output = self.dropout(output)

        # (N, L, H) → (N, L, vocab_size)：每个位置输出对词表的打分
        logits = self.linear(output)

        return logits, hidden

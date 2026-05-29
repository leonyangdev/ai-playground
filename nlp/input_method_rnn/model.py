import torch
import torch.nn as nn

from config import EMBEDDING_DIM, HIDDEN_DIM


class InputMethodModel(nn.Module):
    """基于 RNN 的语言模型（RNNLM），用于输入法下一词预测。

    网络结构（三层串联）：
        Embedding  →  RNN  →  Linear (全连接)

    输入：形状 (N, SEQ_LEN) 的 token id 序列
    输出：形状 (N, vocab_size) 的 logits（未归一化分数）

    为什么取最后一个时间步？
        RNN 在每个时间步更新隐状态，最后一步的隐状态已"看过"全部输入，
        是对整个输入序列最完整的压缩表示，适合用来预测下一个词。
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        # 将离散 token id 映射为连续稠密向量（可学习的查找表）
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_DIM)
        # 单层单向 RNN：batch_first=True 使输入形状为 (N, L, input_size)
        self.rnn = nn.RNN(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_DIM, batch_first=True)
        # 将隐状态映射到词表大小的分类空间
        self.fc = nn.Linear(in_features=HIDDEN_DIM, out_features=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, L) → (N, L, EMBEDDING_DIM)
        embed = self.embedding(x)
        # rnn_out: (N, L, HIDDEN_DIM)；忽略最终隐状态 h_n
        rnn_out, _ = self.rnn(embed)
        # 取最后时间步作为序列的上下文摘要特征
        last_hidden = rnn_out[:, -1, :]   # (N, HIDDEN_DIM)
        logits = self.fc(last_hidden)      # (N, vocab_size)
        return logits


if __name__ == '__main__':
    vocab_size = 1000
    x = torch.randint(vocab_size, size=(64, 5))   # 模拟 batch_size=64，seq_len=5
    model = InputMethodModel(vocab_size)
    output = model(x)
    print(f"输入形状：{x.shape}  输出形状：{output.shape}")

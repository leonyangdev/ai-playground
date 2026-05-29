import jieba

from config import UNK_TOKEN, MODEL_DIR, VOCAB_FILE


class JiebaTokenizer:
    """基于 jieba 的中文分词器，支持词表构建、文本编码与 id 解码。

    使用流程：
        训练前  → build_vocab()  从语料统计词表并写入文件
        训练/推理 → from_vocab()  从文件恢复分词器实例
    """

    # UNK 固定为类属性，确保所有实例共享同一特殊 token
    unk_token = UNK_TOKEN

    def __init__(self, vocab_list: list[str]):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        # 正向映射（编码用）：token → id
        self.word2id: dict[str, int] = {word: idx for idx, word in enumerate(vocab_list)}
        # 反向映射（解码用）：id → token
        self.id2word: dict[int, str] = {idx: word for idx, word in enumerate(vocab_list)}
        self.unk_id = self.word2id[self.unk_token]

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """jieba 精确模式分词，返回 token 列表。"""
        return jieba.lcut(text)

    def encode(self, text: str) -> list[int]:
        """分词后将每个 token 映射为 id；词表外的词使用 unk_id 代替。"""
        tokens = self.tokenize(text)
        return [self.word2id.get(token, self.unk_id) for token in tokens]

    @classmethod
    def build_vocab(cls, sentences: list[str], vocab_file_path) -> None:
        """遍历训练语料，统计所有不重复的 token，写入词表文件。

        只基于训练集构建词表，避免测试集信息提前泄露（数据泄漏问题）。
        UNK 固定排在第 0 位，确保 unk_id == 0。
        """
        vocab_set: set[str] = set()
        for sentence in sentences:
            vocab_set.update(jieba.lcut(sentence))

        vocab_list = [cls.unk_token] + list(vocab_set)
        print(f"词表大小：{len(vocab_list)}")

        with open(vocab_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_file_path) -> 'JiebaTokenizer':
        """从词表文件读取 token 列表，返回初始化好的分词器实例。"""
        with open(vocab_file_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)


if __name__ == '__main__':
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR / VOCAB_FILE)
    print(f"词表大小：{tokenizer.vocab_size}")
    print(f"UNK token：{tokenizer.unk_token}，UNK id：{tokenizer.unk_id}")
    print(f"编码 '自然语言处理'：{tokenizer.encode('自然语言处理')}")

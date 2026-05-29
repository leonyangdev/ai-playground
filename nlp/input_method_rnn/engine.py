"""输入法推理引擎

InputMethodEngine 是整个应用的核心，封装了三件事：
    1. 模型与词表的加载（初始化时一次性完成）
    2. 文本 → 候选词的预测接口
    3. 交互式输入法的运行循环
"""

import torch

from config import MODEL_DIR, VOCAB_FILE, BEST_MODEL_FILE, DEVICE
from model import InputMethodModel
from tokenizer import JiebaTokenizer


class InputMethodEngine:
    """加载训练好的模型，对外暴露候选词预测接口。"""

    def __init__(self):
        self.device = DEVICE

        # 词表大小决定了模型 Embedding 层和输出层的维度，必须先于模型加载
        self.tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR / VOCAB_FILE)

        self.model = InputMethodModel(vocab_size=self.tokenizer.vocab_size).to(self.device)
        self.model.load_state_dict(
            torch.load(MODEL_DIR / BEST_MODEL_FILE, map_location=self.device)
        )
        self.model.eval()   # 切换推理模式，关闭 Dropout 等训练专用行为

    def predict(self, text: str, k: int = 5) -> list[str]:
        """对输入文本预测下一个词的 top-k 候选列表。

        Args:
            text: 已输入的文本（可以是多轮累积的历史输入）
            k:    返回的候选词数量

        Returns:
            按预测概率从高到低排列的候选 token 列表
        """
        token_ids = self.tokenizer.encode(text)
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids)                      # (1, vocab_size)  前向传播

        top_k_ids = torch.topk(logits, k).indices[0].tolist()  # (k,)
        return [self.tokenizer.id2word[idx] for idx in top_k_ids]

    def run(self):
        """启动交互式输入法 demo。"""
        print("欢迎使用智能输入法！输入 q 或 quit 退出...")
        history = ''
        while True:
            user_input = input('> ')
            if user_input.strip() in ('q', 'quit'):
                print("再见！")
                break
            if not user_input.strip():
                print("请输入有效内容！")
                continue

            # 累积历史输入，让模型看到更多上下文来做预测
            history += user_input
            candidates = self.predict(history, k=5)
            print(f"候选词：{candidates}")

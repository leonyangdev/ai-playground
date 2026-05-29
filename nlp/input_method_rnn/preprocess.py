"""数据预处理脚本

原始对话语料 → 句子提取与清洗 → 词表构建 → 滑动窗口样本生成 → 保存 JSONL

只需在训练前运行一次：
    python preprocess.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_DIR, RAW_DATA_FILE,
    PROCESSED_DATA_DIR, TRAIN_DATA_FILE, TEST_DATA_FILE,
    MODEL_DIR, VOCAB_FILE,
    SEQ_LEN,
)
from tokenizer import JiebaTokenizer


def build_sliding_window_samples(sentences: list[str], tokenizer: JiebaTokenizer) -> list[dict]:
    """用滑动窗口将句子列表转换为 (input_ids, target_id) 样本。

    对每个句子的 id 序列，以步长 1 向右滑动：
        token_ids = [w0, w1, w2, w3, w4, w5, ...]
        第 0 条：input=[w0..w4]，target=w5
        第 1 条：input=[w1..w5]，target=w6
        ...

    这是语言模型训练数据的标准构建方式，最大化数据利用率。
    """
    samples = []
    for sentence in sentences:
        token_ids = tokenizer.encode(sentence)
        for i in range(len(token_ids) - SEQ_LEN):
            samples.append({
                'input':  token_ids[i : i + SEQ_LEN],
                'target': token_ids[i + SEQ_LEN],
            })
    return samples


def main():
    print("=" * 40)
    print("开始数据预处理...")
    print("=" * 40)

    # Step 1：读取原始语料（随机抽取 10%，加速实验；正式训练可调为 1.0）
    df = pd.read_json(RAW_DATA_DIR / RAW_DATA_FILE, lines=True, orient='records').sample(frac=0.1)

    # Step 2：提取对话句子并清洗（原始格式 "说话人：内容"，只取冒号后内容）
    sentences = []
    for dialog in df['dialog']:
        for utterance in dialog:
            sentences.append(utterance.split('：')[1])
    print(f"共提取 {len(sentences)} 条句子，示例：{sentences[0]}")

    # Step 3：划分训练集 / 测试集（8:2），固定随机种子保证可复现
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)

    # Step 4：仅基于训练集构建词表（防止测试集词汇提前泄露到词表中）
    JiebaTokenizer.build_vocab(train_sentences, MODEL_DIR / VOCAB_FILE)

    # Step 5：加载词表，初始化分词器
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR / VOCAB_FILE)

    # Step 6：滑动窗口生成训练/测试样本
    train_samples = build_sliding_window_samples(train_sentences, tokenizer)
    test_samples  = build_sliding_window_samples(test_sentences,  tokenizer)
    print(f"训练样本数：{len(train_samples)}  测试样本数：{len(test_samples)}")

    # Step 7：保存为 JSONL（每行一条 JSON，支持流式读取大文件）
    pd.DataFrame(train_samples).to_json(PROCESSED_DATA_DIR / TRAIN_DATA_FILE, orient='records', lines=True)
    pd.DataFrame(test_samples).to_json( PROCESSED_DATA_DIR / TEST_DATA_FILE,  orient='records', lines=True)

    print("数据预处理完成！")


if __name__ == '__main__':
    main()

import torch
from config import BATCH_SIZE, PROCESSED_DATA_DIR, TEST_DATA_FILE, TRAIN_DATA_FILE
from torch.utils.data import DataLoader, Dataset

import pandas as pd


class InputMethodDataset(Dataset):
    """从预处理好的 JSONL 文件中加载 (input_ids, target_id) 样本对。

    每行 JSONL 格式：{"input": [id0, ..., id_{SEQ_LEN-1}], "target": id_{SEQ_LEN}}
    """

    def __init__(self, path):
        # 读取 JSONL 文件（每行一个 JSON 对象），将其转换为字典列表。
        # lines=True: 指定文件格式为 JSON Lines，即每一行是一个独立的 JSON 对象。
        # orient="records": 将 DataFrame 的每一行解析为一个字典，键为列名。
        # .to_dict(orient="records"): 最终将 DataFrame 转换为列表，列表中每个元素是对应行的字典。
        # 结果示例: [{"input": [...], "target": ...}, {"input": [...], "target": ...}, ...]
        self.samples = pd.read_json(path, lines=True, orient="records").to_dict(
            orient="records"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        input_ids = torch.tensor(sample["input"], dtype=torch.long)
        target_id = torch.tensor(sample["target"], dtype=torch.long)
        return input_ids, target_id


def get_dataloader(split: str = "train") -> DataLoader:
    """返回训练集或测试集的 DataLoader。

    Args:
        split: 'train' 或 'test'

    训练集开启 shuffle 以打破样本顺序相关性；测试集保持原序保证评估可复现。
    """
    file_map = {"train": TRAIN_DATA_FILE, "test": TEST_DATA_FILE}
    path = PROCESSED_DATA_DIR / file_map[split]
    dataset = InputMethodDataset(path)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(split == "train"))


if __name__ == "__main__":
    train_loader = get_dataloader("train")
    test_loader = get_dataloader("test")

    input_ids, target_id = next(iter(train_loader))
    print(f"input_ids 形状：{input_ids.shape}  target_id 形状：{target_id.shape}")
    print(f"训练集批次数：{len(train_loader)}  测试集批次数：{len(test_loader)}")

from pathlib import Path

# ── 目录结构 ──────────────────────────────────────────────────────────────────
ROOT_DIR           = Path(__file__).parent  # config.py 位于项目根目录
RAW_DATA_DIR       = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'
MODEL_DIR          = ROOT_DIR / 'models'
LOG_DIR            = ROOT_DIR / 'logs'

# ── 文件名 ────────────────────────────────────────────────────────────────────
RAW_DATA_FILE   = 'synthesized_.jsonl'
TRAIN_DATA_FILE = 'train.jsonl'
TEST_DATA_FILE  = 'test.jsonl'
VOCAB_FILE      = 'vocab.txt'       # 每行一个 token，第 0 行为 UNK
BEST_MODEL_FILE = 'best_model.pt'   # 保存训练损失最低时的模型权重

# ── 特殊 Token ────────────────────────────────────────────────────────────────
# 词表外的词（未登录词）统一映射为 UNK，避免 KeyError，保证编码鲁棒性
UNK_TOKEN = '<unk>'

# ── 数据超参数 ────────────────────────────────────────────────────────────────
# 滑动窗口大小：用前 SEQ_LEN 个 token 预测下一个 token（语言模型的核心设定）
SEQ_LEN    = 5
BATCH_SIZE = 64

# ── 模型超参数 ────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 128   # 词向量维度：将离散 token id 映射为稠密向量
HIDDEN_DIM    = 256   # RNN 隐藏层维度：越大表达能力越强，但训练越慢

# ── 训练超参数 ────────────────────────────────────────────────────────────────
LEARNING_RATE = 1e-3
EPOCHS        = 10

# ── 设备选择 ──────────────────────────────────────────────────────────────────
import torch
# 优先级：CUDA（NVIDIA）> MPS（Apple Silicon）> CPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

"""智能输入法 — 应用入口

直接运行本文件即可启动输入法：
    python main.py

模型训练相关脚本：
    python preprocess.py   # 数据预处理（训练前运行一次）
    python train.py        # 训练模型
    python evaluate.py     # 评估模型准确率
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import InputMethodEngine

if __name__ == '__main__':
    InputMethodEngine().run()

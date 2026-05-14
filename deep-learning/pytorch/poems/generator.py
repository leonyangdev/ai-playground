"""
诗歌生成模块（推理/预测）
使用训练好的语言模型，逐字自回归地生成古诗。

自回归生成（Auto-regressive Generation）原理：
    每次只预测一个字，再将其作为下一步的输入，循环往复，
    直到生成完整的诗——这与人类逐字写诗的方式高度类似。
"""
import torch


def generate_poem(model, id2word, word2id, start_token,
                  line_num=4, line_len=7, temperature=1.0, device='cpu'):
    """
    逐字自回归生成一首古诗。

    关键设计说明：
      - hidden state 持续传递：hidden 在整首诗的生成过程中不断更新，
        使模型在生成每个字时都能"记住"已经生成过的内容（上下文记忆）。
        原始代码将 hidden 丢弃（`_ = model(input)`），导致每个字的生成
        都从空白上下文开始，这里已修复。
      - temperature 采样：用温度系数缩放 logits，再做 softmax，
        temperature < 1 → 分布变尖，倾向高概率字（生成更"正统"）
        temperature > 1 → 分布变平，随机性增加（生成更"有创意"）
      - multinomial 采样：不取 argmax（贪心），而是按概率随机采样，
        同一个起始字每次运行都能生成不同的诗。

    Args:
        model:       训练好的 PoemRNNLM 模型
        id2word:     ID → 字 的映射列表
        word2id:     字 → ID 的映射字典
        start_token: 起始字（可为任意汉字）
        line_num:    生成的行数（每行含"上句，下句。"各 line_len 字）
        line_len:    每句字数（五言=5，七言=7）
        temperature: 采样温度（默认 1.0，即使用原始概率分布）
        device:      计算设备（需与模型所在设备一致）

    Returns:
        poem_str: 生成的完整古诗字符串（含标点换行）
    """
    model.eval()

    unk_id = word2id["<UNK>"]
    poem_chars = []   # 按字符顺序记录生成结果（含标点）

    # 起始 token 处理：若不在词表中退化为 <UNK>
    start_id = word2id.get(start_token, unk_id)
    if start_id != unk_id:
        poem_chars.append(start_token)
        remaining = line_len - 1   # 起始字已占一个位置
    else:
        remaining = line_len

    # 输入张量：(batch=1, seq=1)，从单个字开始逐步生成
    current_input = torch.LongTensor([[start_id]]).to(device)
    hidden = None   # None → RNN 自动以全零初始化隐藏状态

    with torch.no_grad():   # 推理阶段无需计算梯度，节省显存和计算量
        for _ in range(line_num):
            # 每行包含两句：上句结尾加"，"，下句结尾加"。\n"
            for punctuation in ["，", "。\n"]:

                while remaining > 0:
                    # 前向传播，同时传入并接收隐藏状态
                    # 关键：hidden 携带了之前所有字的上下文信息
                    logits, hidden = model(current_input, hidden)

                    # 取序列最后一个时间步的输出：(vocab_size,)
                    last_logit = logits[0, -1]

                    # Temperature 缩放：调节采样分布的"锐利程度"
                    proba = torch.softmax(last_logit / temperature, dim=-1)

                    # 按概率随机采样（非贪心，保证生成多样性）
                    next_id = torch.multinomial(proba, num_samples=1)

                    poem_chars.append(id2word[next_id.item()])

                    # 将新生成的字作为下一步的输入，形状 (1, 1)
                    current_input = next_id.unsqueeze(0)
                    remaining -= 1

                # 当前句结束，追加标点，重置下一句的字数计数
                poem_chars.append(punctuation)
                remaining = line_len

    return "".join(poem_chars)

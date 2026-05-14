import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import re

# 1. 数据预处理，传入语料库文件路径
def pre_process(file_path):
    poems = []      # 按行保存每一首诗
    poems_id = []  # 保存id化后的诗
    char_set = set()    # 保存所有不重复的词（字）
    # 1。1 读取文件，保存诗的内容
    with open(file_path, 'r', encoding='utf-8') as f:
        # 逐行处理，一行就是一首诗
        for line in f:
            # 使用正则去除标点符号和空白
            line = re.sub(r"[，。、？！：]", "", line).strip()
            # 按字分割并去重
            char_set.update(list(line))
            # 按行保存诗（一首）
            poems.append(list(line))

    # print(poems[0])

    # 1.2 构建词表
    # 构建id到word的映射列表
    id2word = list(char_set) + ["<UNK>"]
    # 构建word到id的映射字典
    word2id = { word:id for id, word in enumerate(id2word) }

    # print(id2word)
    # print(word2id)

    # 1.3 语料ID化
    for poem in poems:
        # 对每首诗，将每一个字转换为id，并构成列表
        poem_id = [ word2id.get(word) for word in poem ]
        poems_id.append(poem_id)

    # print(poems_id[0])

    return poems_id, id2word, word2id

poems_id, id2word, word2id = pre_process('./data/poems.txt')

print(len(poems_id), len(id2word), len(word2id))

# 2. 创建数据集
# 自定义DataSet类，元素 （x, y）都是长度为 L的文本id序列
class PoemDataset(Dataset):
    # 初始化
    def __init__(self, poems_id, seq_len):
        self.seq_len = seq_len
        self.dataset = []   # 保存训练数据集，元组(x, y)的列表
        # 遍历id化的语料库，处理每首诗
        for poem_id in poems_id:
            # 遍历每首诗中的每个字（id）
            for i in range(len(poem_id) - self.seq_len):
                x = poem_id[i: i+self.seq_len]  # 取长度为 L 的序列作为 x、y
                y = poem_id[i+1: i+1+self.seq_len]
                self.dataset.append((x, y))
    # 获取长度
    def __len__(self):
        return len(self.dataset)
    # 按索引号获取元素
    def __getitem__(self, idx):
        # 将x、y包装成张量返回
        x = torch.LongTensor(self.dataset[idx][0])
        y = torch.LongTensor(self.dataset[idx][1])
        return x, y

dataset = PoemDataset(poems_id, seq_len=24)
print(len(dataset))
# print(dataset[0])

# 3. 搭建模型
# 自定义RNNLM模型类
class PoemRNNLM(nn.Module):
    # 初始化
    def __init__(self, vocab_size, embedding_size=128, hidden_size=256, num_layers=1):
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        # RNN
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # 全连接层（输出层）
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
    # 前向传播
    def forward(self, input, hx = None):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hx)
        output = self.linear(output)
        return output, hidden

model = PoemRNNLM(vocab_size=len(id2word), embedding_size=256, hidden_size=512, num_layers=2)

# 4. 初始化相关操作
# 4.1 加载到设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4.2 定义超参数
lr = 1e-3
batch_size = 32
epoch_num = 20

# 4.3 定义优化器
optimizer = optim.Adam(model.parameters(), lr=lr)

# 4.4 定义损失函数
loss = nn.CrossEntropyLoss()

# 4.5 定义数据加载器
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. 模型训练
# 5.1 设置训练模式
model.train()

# 5.2 迭代轮次，进行本轮训练
for epoch in range(epoch_num):
    total_loss = 0  # 累计训练损失
    # 当前轮次，按批次依次迭代训练，x、y 形状为（N, L）
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        # 前向传播
        output, _ = model(x)
        # 计算损失，需要将预测形状调整为（N, C, L）
        loss_value = loss(output.transpose(1, 2), y)
        # 反向传播，计算梯度
        loss_value.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

        # 累加损失
        total_loss += loss_value.item() * x.shape[0]

        # 打印进度条
        print(f"\repoch: {epoch + 1:0>2}[{'=' * int((i + 1) / len(dataloader) * 50):<50}]", end='')
    # 5.3 本轮训练完毕
    this_loss = total_loss / len(dataset)
    print(f"train loss: {this_loss:.6f}")

# 6. 生成新诗（推理/预测）
def generate_poem(model, id2word, word2id, start_token, line_num=4, length=7):
    # 6.1 设置验证模型
    model.eval()

    poem = []   # 记录生成诗的结果，字符数组
    current_len = length

    # 6.2 将起始token转换成id
    start_id = word2id.get(start_token, word2id["<UNK>"])
    # 如果起始词在词表中，就添加到生成的诗中
    if start_id != word2id["<UNK>"]:
        poem.append(start_token)
        current_len -= 1

    # 6.3 定义输入数据，包装成(N, L)形状的张量
    input = torch.LongTensor([[start_id]]).to(device)

    # 6.4 推理，生成诗句
    with torch.no_grad():
        # 遍历行数，按行生成诗
        for i in range(line_num):
            # 遍历当前行的两句，加上不同的标点符号
            for interpunction in ["，", "。\n"]:
                # 遍历当前句的每个字，前向传播预测生成下一个字
                while current_len > 0:
                    # 前向传播，得到分类输出
                    output, _ = model(input)
                    # 得到分类概率
                    proba = torch.softmax(output[0, 0], dim=-1)
                    # 按概率随机选取id
                    next_id = torch.multinomial(proba, num_samples=1)
                    # 转换成汉字保存到诗中
                    poem.append( id2word[next_id.item()] )
                    # 更新input，继续生成下一个字
                    input = next_id.unsqueeze(0)
                    current_len -= 1
                # 本句诗已生成，加上标点，开始下一句
                poem.append(interpunction)
                current_len = length
    # 全部诗生成完毕
    return "".join(poem)

# 生成 10 首诗
for i in range(10):
    print(generate_poem(model, id2word, word2id, start_token="一", line_num=4, length=7))
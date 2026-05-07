import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader


# =====================================================================
# 1. 数据准备与预处理函数
# =====================================================================
def create_dataset(data_path):
  """
  读取原始 CSV 数据，并进行特征填充、标准化、独热编码，最终打包为 PyTorch 的 Dataset。
  """
  # 读取数据
  data = pd.read_csv(data_path)
  print("数据集原始列数：", data.shape[1])

  # 去除与预测房价无关的 ID 列
  data.drop(["Id"], axis=1, inplace=True)

  # 划分特征矩阵 X 和目标变量 y (房屋售价)
  X = data.drop(["SalePrice"], axis=1)
  y = data["SalePrice"]

  print("特征矩阵大小：", X.shape)

  # 【优化】更严谨地筛选特征类型
  # "number" 会自动匹配 int32, int64, float32, float64 等所有数值型
  numerical_features = X.select_dtypes(include=["number"]).columns
  # 类别型特征同时包容传统的 object、Pandas3.0 新的 string 以及特殊的 category 类型
  categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns

  print(f"数值型特征数量: {len(numerical_features)} | 类别型特征数量: {len(categorical_features)}")

  # 划分训练集和测试集（80% 训练，20% 测试）
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # 构建数值型特征的处理流水线（Pipeline）
  # 步骤 1: 用中位数/平均值填充缺失值
  # 步骤 2: 将数据缩放到均值为 0，方差为 1 的标准正态分布，消除量纲影响，加速神经网络收敛
  numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
  ])

  # 构建类别型特征的处理流水线
  # 步骤 1: 类别特征缺失值填充为 "NaN" 字符串，将其作为独立的一个类别处理
  # 步骤 2: 独热编码。handle_unknown="ignore" 极其重要，能避免测试集出现训练集没见过的未知类别时崩溃
  categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="NaN")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
  ])

  # 使用 ColumnTransformer 将上述两种不同的流水线组合起来，分别作用于不同的列
  preprocessor = ColumnTransformer(
    transformers=[
      ("numerical", numerical_transformer, numerical_features),
      ("categorical", categorical_transformer, categorical_features)
    ]
  )

  # 【核心步骤】拟合并转换数据
  # fit_transform: 在训练集上“学习”均值、标准差、类别字典，并进行转换
  # toarray(): 将 One-Hot 编码后产生的高维稀疏矩阵(Sparse Matrix)还原成标准的密集二维数组
  x_train_arr = preprocessor.fit_transform(X_train).toarray()
  # transform: 在测试集上做转换时，必须直接复用训练集学到的规则，严禁使用 fit！避免数据泄露。
  x_test_arr = preprocessor.transform(X_test).toarray()

  # 将转换后的 NumPy 二维数组重新组装成带有列名的 DataFrame，方便预览和保存
  feature_names = preprocessor.get_feature_names_out()
  x_train_df = pd.DataFrame(x_train_arr, columns=feature_names)
  x_test_df = pd.DataFrame(x_test_arr, columns=feature_names)

  # 包装成 PyTorch 专用的 TensorDataset 格式
  # values 将 DataFrame 转为 NumPy 数组，再通过 torch.tensor 转为 PyTorch 张量
  train_dataset = TensorDataset(
    torch.tensor(x_train_df.values, dtype=torch.float32),
    torch.tensor(y_train.values, dtype=torch.float32)
  )
  test_dataset = TensorDataset(
    torch.tensor(x_test_df.values, dtype=torch.float32),
    torch.tensor(y_test.values, dtype=torch.float32)
  )

  # 导出预览文件，用于特征分析
  x_train_df.to_csv("data/x_train_preview.csv", index=False)

  return train_dataset, test_dataset, x_train_df.shape[1]


# =====================================================================
# 2. 损失函数 (RMSLE - 对数均方根误差)
# =====================================================================
def loss_fn(y_pred, y_true):
  r"""
  计算对数均方根误差 (RMSLE)。
  这是房价预测这类偏态分布（价格跨度极大）任务的标准评估指标。
  公式为: $RMSLE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\ln(y_{pred}) - \ln(y_{true}))^2}$
  """
  mse = nn.MSELoss()
  
  # 降维：将形状从 [Batch, 1] 压缩为 [Batch]，防止计算 MSE 时因维度不一致引发广播机制错误
  y_pred = y_pred.squeeze(-1)
  y_true = y_true.squeeze()

  # 裁剪预测值：因为 log(x) 在 x<=0 时无意义。
  # 限制预测值最小为 1，确保取 log 后的数值是安全、合法的，避免产生 NaN
  y_pred = torch.clamp(y_pred, 1.0, float("inf"))
  y_true = torch.clamp(y_true, 1.0, float("inf"))

  # 计算 log 之后的均方误差，并开根号
  return torch.sqrt(mse(torch.log(y_pred), torch.log(y_true)))


# =====================================================================
# 3. 神经网络模型训练主循环
# =====================================================================
def train(model, train_dataset, test_dataset, lr, epochs, batch_size, device):
  # 1. 权重初始化函数：为全连接层(Linear)使用 Xavier 均匀分布初始化，偏置初始化为 0
  # 这能保证前向传播时各层激活值的方差保持一致，避免梯度消失/爆炸
  def init_weights(m):
    if isinstance(m, nn.Linear):
      nn.init.xavier_normal_(m.weight)
      nn.init.zeros_(m.bias)

  model.apply(init_weights)
  model = model.to(device)

  # 2. 定义优化器
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  # 【优化】在 Epoch 循环外初始化好 DataLoader，避免重复创建的系统开销
  # shuffle=True: 每次 Epoch 开始前打乱数据，打破样本顺序依赖，防止模型“死记硬背”
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  train_loss_list = []
  test_loss_list = []

  # 开始 Epoch 大循环
  for epoch in range(epochs):
    # ------------------ 训练阶段 (Training) ------------------
    model.train()  # 激活 BatchNorm 和 Dropout 机制
    train_loss_accumulate = 0.0
    total_train_samples = 0

    for X_batch, y_batch in train_loader:
      X_batch, y_batch = X_batch.to(device), y_batch.to(device)

      # 前向传播：将特征输入模型，得到预测房价
      output = model(X_batch)

      # 计算当前 Batch 的损失
      loss = loss_fn(output, y_batch)

      # 反向传播三步：
      optimizer.zero_grad()  # 清空上一次累积的梯度
      loss.backward()  # 反向传播计算当前梯度
      optimizer.step()  # 根据优化算法（Adam）更新网络权重

      # 累加 Loss：通过 loss.item() 乘以当前 Batch 的实际样本数
      train_loss_accumulate += loss.item() * X_batch.size(0)
      total_train_samples += X_batch.size(0)

    # 计算当前 Epoch 的平均训练损失并记录
    epoch_train_loss = train_loss_accumulate / total_train_samples
    train_loss_list.append(epoch_train_loss)

    # ------------------ 验证阶段 (Evaluation) ------------------
    model.eval()  # 关闭 BatchNorm 的均值更新，关闭 Dropout 的丢弃行为，使预测稳定
    test_loss_accumulate = 0.0
    total_test_samples = 0

    # 关闭梯度计算，节省内存，提升推理速度
    with torch.no_grad():
      for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        output = model(X_batch)
        loss = loss_fn(output, y_batch)

        # 累加测试 Loss
        test_loss_accumulate += loss.item() * X_batch.size(0)
        total_test_samples += X_batch.size(0)

    # 计算当前 Epoch 的平均测试损失并记录
    epoch_test_loss = test_loss_accumulate / total_test_samples
    test_loss_list.append(epoch_test_loss)

    # 打印当前 Epoch 的训练概况（每隔 10 个 Epoch 打印一次，避免刷屏）
    if (epoch + 1) % 10 == 0 or epoch == 0:
      print(f"Epoch [{epoch + 1:03d}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f}")

  return train_loss_list, test_loss_list


# =====================================================================
# 4. 主函数
# =====================================================================
def main():
  # 1. 创建数据集
  train_dataset, test_dataset, feature_size = create_dataset("data/train.csv")

  print("\n--- 数据集准备就绪 ---")
  print("训练集样本数：", len(train_dataset))
  print("测试集样本数：", len(test_dataset))
  print("输入模型特征维度：", feature_size)

  # 2. 搭建多层感知机模型 (MLP)
  # 对于表格数据，BN + Dropout 是对抗过拟合、加速收敛的黄金组合

  model = nn.Sequential(
    # 第一层：全连接，将高维 One-Hot 特征映射到 128 维空间
    nn.Linear(feature_size, 128),
    # 批归一化：在每批数据进入激活函数前进行标准化，防止内部协变量偏移，使模型对初始化和学习率更鲁棒
    nn.BatchNorm1d(128),
    # 激活函数：引入非线性表达能力
    nn.ReLU(),
    # 随机丢弃：训练时随机让 20% 的神经元失活，强迫网络学习鲁棒特征，极大缓解过拟合
    nn.Dropout(0.2),
    # 输出层：将 128 维映射到 1 维（预测的房价标量）
    nn.Linear(128, 1),
  )

  # 3. 硬件加速检测

  # 原来的逻辑：
  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # 针对 Mac 优化的逻辑：
  if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("检测到 Apple Silicon GPU，已开启 MPS 加速。")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("检测到 NVIDIA GPU，已开启 CUDA 加速。")
  else:
    device = torch.device("cpu")
    print("未检测到 GPU 加速硬件，使用 CPU 运行。")
  print(f"当前运行设备: {device}")

  # 4. 开始训练
  # 适当降低 Epoch 数量或根据优化后的曲线调整超参数
  train_loss_list, test_loss_list = train(
    model,
    train_dataset,
    test_dataset,
    lr=0.01,
    epochs=500,
    batch_size=64,
    device=device
  )

  # 5. 绘制平滑正确的训练曲线
  plt.figure(figsize=(10, 6))
  plt.plot(train_loss_list, "r-", label="Train Loss (RMSLE)", linewidth=2.5)
  plt.plot(test_loss_list, "b--", label="Test Loss (RMSLE)", linewidth=2)
  plt.title("Training and Testing Loss Curve")
  plt.xlabel("Epochs")
  plt.ylabel("Loss (RMSLE)")
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.show()


if __name__ == "__main__":
  main()
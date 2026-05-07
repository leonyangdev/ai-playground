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


def create_dataset(data_path):
  # 读取数据
  data = pd.read_csv(data_path)

  # 查看表格列数
  print("数据集列数：", data.shape[1])

  # 数据预处理,去除无关特征
  data.drop(["Id"], axis=1, inplace=True)

  # 划分特征和目标
  X = data.drop(["SalePrice"], axis=1)
  y = data["SalePrice"]

  # print(data.head())
  print("数据集大小：", X.shape)

  # 筛选出数值型特征
  numerical_features = X.select_dtypes(exclude="object").columns

  # 筛选出类别型特征
  categorical_features = X.select_dtypes(include=["object", "string"]).columns

  print("数值型特征：", numerical_features)
  print("类别型特征：", categorical_features)

  # 数据集划分
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # 特征预处理

  # 数值型特征先用平均值填充缺失值，再进行标准化
  numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
  ])

  # 类别型特征先将缺失值替换为字符串"NaN"，再进行独热编码
  categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="NaN")),
    # 在独热编码里，设置`ignore`后，遇到没在训练数据里出现过的新类别，就把对应编码都设为0，不报错。 
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
  ])

  # 组合特征预处理
  preprocessor = ColumnTransformer(
    transformers=[
      ("numerical", numerical_transformer, numerical_features),
      ("categorical", categorical_transformer, categorical_features)
    ]
  )

  # 进行特征预处理
  # 在执行 toarray() 之前，数据是稀疏矩阵，只记录了非 0 元素的位置，不能直接给 Dataframe, 需要转换为二维数组, 才能转换为 Dataframe
  """
  转换前：
    (0, 0)	1
  (1, 1)	1
  (2, 0)	1
  
  转换后：
  [[1, 0, 0],
   [0, 1, 0],
   [1, 0, 0]]
  """
  x_train = pd.DataFrame(preprocessor.fit_transform(X_train).toarray(), columns=preprocessor.get_feature_names_out())
  x_test = pd.DataFrame(preprocessor.transform(X_test).toarray(), columns=preprocessor.get_feature_names_out())
  print(x_train)

  # 构建数据集
  train_dataset = TensorDataset(torch.tensor(x_train.values, dtype=torch.float32),
                                torch.tensor(y_train.values, dtype=torch.float32))

  test_dataset = TensorDataset(torch.tensor(x_test.values, dtype=torch.float32),
                               torch.tensor(y_test.values, dtype=torch.float32))

  # 将 x_train 输出到 data/x_train_preview.csv
  x_train.to_csv("data/x_train_preview.csv", index=False)

  return train_dataset, test_dataset, x_train.shape[1]


# 损失函数
def loss_fn(y_pred, y_true):
  mse = nn.MSELoss()

  y_pred.squeeze_()

  y_pred = torch.clamp(y_pred, 1, float("inf"))

  return torch.sqrt(mse(torch.log(y_pred), torch.log(y_true)))


def train(model, train_dataset, test_dataset, lr, epochs, batch_size, device):
  def init_weights(m):
    if type(m) == nn.Linear:
      nn.init.xavier_normal_(m.weight)
      nn.init.zeros_(m.bias)

  model.apply(init_weights)

  model = model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  train_loss_list = []
  test_loss_list = []

  for epoch in range(epochs):
    # 训练过程
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loss_accumlate = 0

    for batch_count, (X, y) in enumerate(train_loader):
      X = X.to(device)
      y = y.to(device)

      # 前向传播
      output = model(X)

      # 反向传播
      loss = loss_fn(output, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # 累加损失
      train_loss_accumlate += loss.item()

      # 打印训练进度
      print(f"Epoch: {epoch + 1}/{epochs}, Batch: {batch_count + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

      # 计算平均损失
      this_train_loss = train_loss_accumlate / len(train_loader)
      train_loss_list.append(this_train_loss)

    # 验证过程
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss_accumulate = 0

    with torch.no_grad():
      for (X, y) in test_loader:
        X = X.to(device)
        y = y.to(device)

        output = model(X)

        loss = loss_fn(output, y)

        test_loss_accumulate += loss.item()

        this_test_loss = test_loss_accumulate / len(test_loader)
        test_loss_list.append(this_test_loss)

    # 打印训练损失，验证损失
    print(f"Epoch: {epoch + 1}/{epochs}, Train Loss: {this_train_loss:.4f}, Test Loss: {this_test_loss:.4f}")

  return train_loss_list, test_loss_list


def main():
  train_dataset, test_dataset, feature_size = create_dataset("data/train.csv")

  print("训练集大小：", len(train_dataset))
  print("测试集大小：", len(test_dataset))
  print("输入特征维度：", feature_size)  # 这个相比于原始 csv 的 81 列多了很多列，因为类别型特征被独热编码了，增加了列

  # 搭建模型
  model = nn.Sequential(
    nn.Linear(feature_size, 128),
    # 对输入的 128 维数据，按批次计算均值和方差，再进行归一化，可加速收敛，减少内部协变量偏移，提升模型稳定性与泛化能力。
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1),
  )

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  train_loss_list, test_loss_list = train(model, train_dataset, test_dataset, lr=0.01, epochs=200, batch_size=64,
                                          device=device)
  
  plt.plot(train_loss_list, "r-", label="Train Loss", linewidth=3)
  
  plt.plot(test_loss_list, "k--", label="Test Loss", linewidth=2)
  
  plt.legend()
  plt.show()


if __name__ == "__main__":
  main()

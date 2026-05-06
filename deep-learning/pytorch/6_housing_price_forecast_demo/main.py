import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import TensorDataset


def create_dataset(data_path):
  # 读取数据
  data = pd.read_csv(data_path)

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
    # 忽略独热编码时遇到的未知类别，使新类别编码为全0
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

  return train_dataset, test_dataset, x_train.shape[1]


def main():
  train_dataset, test_dataset, input_size = create_dataset("data/train.csv")

  print("训练集大小：", len(train_dataset))
  print("测试集大小：", len(test_dataset))
  print("输入特征维度：", input_size)

if __name__ == "__main__":
  main()

# 第 6 章：反向传播

> 💡 **导读：** 这是整本小册子最核心的章节。反向传播是神经网络训练的引擎，我们将从最末端的损失函数出发，一步步反向推导，带你领略数学公式中奇妙的“抵消与坍缩”之美。

## 6.1 问题：参数该如何更新？（量化推导）

训练的终极目标是让损失函数 $L$ 的值尽可能小。我们目前有 39,760 个参数，每次前向传播计算出损失后，我们需要弄清楚：**该把每个参数调大一点，还是调小一点？**

这里我们需要用微积分中的**一阶泰勒展开式**来进行量化。假设我们给参数 $\theta$ 施加一个极小的变化量 $\Delta \theta$，那么新的损失可以近似表示为：
$$L(\theta + \Delta \theta) \approx L(\theta) + \Delta \theta \cdot \frac{\partial L}{\partial \theta}$$

我们的目标是让新的损失变小，即 $L(\theta + \Delta \theta) < L(\theta)$。这就要求公式右边的尾巴必须是负数：
$$\Delta \theta \cdot \frac{\partial L}{\partial \theta} < 0$$

这就得出了参数更新的铁律：**变化量 $\Delta \theta$ 的符号，必须和梯度 $\frac{\partial L}{\partial \theta}$ 的符号相反！**
* **如果梯度 > 0**：为了让乘积小于 0，$\Delta \theta$ 必须为负（即**减小**参数）。
* **如果梯度 < 0**：为了让乘积小于 0，$\Delta \theta$ 必须为正（即**增大**参数）。



反向传播算法，就是用来极其高效地计算出所有参数对应的梯度 $\frac{\partial L}{\partial \theta}$。

---

## 6.2 计算图与链式法则

我们的神经网络前向传播是一条严密的流水线（从左到右）：
> 输入 $X \rightarrow$ [线性输出 $a_1$] $\rightarrow$ [激活输出 $z_1$] $\rightarrow$ [线性输出 $a_2$] $\rightarrow$ [激活输出 $y$] $\rightarrow$ [计算损失 $L$]

反向传播则是时光倒流，利用**链式法则**，从最右边的 $L$ 逐级向左求导：

$$ L \rightarrow \frac{\partial L}{\partial y} \rightarrow \frac{\partial L}{\partial a_2} \rightarrow \frac{\partial L}{\partial z_1} \rightarrow \frac{\partial L}{\partial a_1} \rightarrow  \frac{\partial L}{\partial W_1} $$

---

## 6.3 符号约定

为了让接下来的推导清晰无负担，我们约定以下符号。**黄金规律：任何一层变量的梯度矩阵，其形状（Shape）必定与原变量完全一致。**


| 变量含义 | 梯度简写 | 数学表达 | 矩阵形状 (Shape) |
| :--- | :--- | :--- | :--- |
| 第二层激活输出 (预测概率) | $dy_{out}$ | $\frac{\partial L}{\partial y}$ | `(n, 10)` |
| 第二层线性输出 (Logits) | $da_2$ | $\frac{\partial L}{\partial a_2}$ | `(n, 10)` |
| 第一层激活输出 | $dz_1$ | $\frac{\partial L}{\partial z_1}$ | `(n, 50)` |
| 第一层线性输出 | $da_1$ | $\frac{\partial L}{\partial a_1}$ | `(n, 50)` |
| 第二层权重参数 | $dW_2$ | $\frac{\partial L}{\partial W_2}$ | `(50, 10)` |
| 第二层偏置参数 | $db_2$ | $\frac{\partial L}{\partial b_2}$ | `(10,)` |
| 第一层权重参数 | $dW_1$ | $\frac{\partial L}{\partial W_1}$ | `(784, 50)` |
| 第一层偏置参数 | $db_1$ | $\frac{\partial L}{\partial b_1}$ | `(50,)` |

*(注：$n$ 为批量样本大小，推导时我们先按单个样本推导，最后求平均)*

---

## 6.4 第一步：损失对激活输出求导 ($L \rightarrow y$)

我们要弄清楚：交叉熵的导数 $\frac{\partial \ell}{\partial y_t} = -\frac{1}{y_t}$ 到底是怎么来的？

首先，我们给出**多分类交叉熵损失函数（Cross-Entropy Loss）的通用标准公式**。
对于单个样本，假设网络总共有 $C$ 个类别（在我们的数字识别里 $C=10$）。$y_j$ 是网络预测的第 $j$ 个类别的概率，$t_j$ 是该类别的真实标签。公式为：
$$\ell = - \sum_{j=0}^{C-1} t_j \log(y_j)$$

**化简过程（One-Hot 编码的魔法）：**
这里的真实标签 $\mathbf{t}$ 是一个 One-Hot 向量。这意味着，除了正确的那个类别（假设索引为 $t$）对应的值 $t_t = 1$ 之外，其他所有错误类别对应的值全是 $0$。
当你把这个 One-Hot 向量代入求和公式时，所有等于 $0$ 的项全部灰飞烟灭了：
$$\ell = - \left( 0 \cdot \log(y_0) + \dots + 1 \cdot \log(y_t) + \dots + 0 \cdot \log(y_9) \right)$$
所以，标准公式直接坍缩成了我们在书里常用的极简形式：
$$\ell = -\log(y_t)$$

**求导过程：**
现在，我们要看这个损失 $\ell$ 是如何随着网络的预测概率 $\mathbf{y}$ 而变化的。
微积分里有一个基础公式：$\frac{d}{dx}\log(x) = \frac{1}{x}$。

1.  **当求导目标是正确类别的概率 $y_t$ 时：**
    公式 $\ell = -\log(y_t)$ 里刚好有 $y_t$，直接套用对数求导公式，保留外面的负号：
    $$\frac{\partial \ell}{\partial y_t} = -\frac{1}{y_t}$$

2.  **当求导目标是其他错误类别的概率 $y_j$ 时 ($j \neq t$)：**
    公式 $\ell = -\log(y_t)$ 里根本没有 $y_j$ 这个变量！对于 $y_j$ 来说，整项 $-\log(y_t)$ 就像一个常数。常数求导等于 $0$：
    $$\frac{\partial \ell}{\partial y_j} = 0$$

这就是第一步梯度的完整由来，没有任何跳跃。这构成了我们向后传递的第一级梯度向量 $\frac{\partial \ell}{\partial \mathbf{y}}$，为后面与 Softmax 的雅可比矩阵相乘做好了准备。

## 6.5 第二步：激活输出对线性输出求导 ($y \rightarrow a_2$) 

接下来，梯度要穿过 Softmax 函数，追溯到第二层的线性输出 $\mathbf{a}_2$。
Softmax 的公式为：$y_j = \frac{e^{a_j}}{\sum_k e^{a_k}}$。

在这里，输入是一个长度为 10 的向量 $\mathbf{a}_2$，输出也是一个长度为 10 的向量 $\mathbf{y}$。在微积分中，向量对向量求导，会得到一个 $10 \times 10$ 的**雅可比矩阵（Jacobian Matrix）** $J$。矩阵中的每一个元素就是 $\frac{\partial y_j}{\partial a_s}$。

为了求这个导数，我们使用商的求导法则：$(\frac{u}{v})' = \frac{u'v - uv'}{v^2}$。
我们令分母 $S = \sum_k e^{a_k}$。无论对哪个 $a_s$ 求导，分母 $S$ 的导数始终是 $e^{a_s}$。

[为什么分两种情况？](./explains/06_jacobian_matrix.md)

**情况 A：当 $j = s$ 时（对角线元素）**
此时分子 $u = e^{a_s}$，分子求导 $u' = e^{a_s}$。
$$\frac{\partial y_s}{\partial a_s} = \frac{(e^{a_s})' \cdot S - e^{a_s} \cdot (S)'}{S^2} = \frac{e^{a_s} \cdot S - e^{a_s} \cdot e^{a_s}}{S^2}$$
将其拆分并化简：
$$= \frac{e^{a_s}}{S} - \left(\frac{e^{a_s}}{S}\right)^2 = y_s - (y_s)^2 = \mathbf{y_s(1 - y_s)}$$

**情况 B：当 $j \neq s$ 时（非对角线元素）**
此时分子 $u = e^{a_j}$，因为它不包含 $a_s$，所以分子求导 $u' = 0$。
$$\frac{\partial y_j}{\partial a_s} = \frac{(e^{a_j})' \cdot S - e^{a_j} \cdot (S)'}{S^2} = \frac{0 \cdot S - e^{a_j} \cdot e^{a_s}}{S^2}$$
将其拆分并化简：
$$= - \left(\frac{e^{a_j}}{S}\right) \cdot \left(\frac{e^{a_s}}{S}\right) = \mathbf{-y_j y_s}$$

至此，我们得到了完整的雅可比矩阵 $J$。

---

## 6.6 见证奇迹：损失对线性输出的联合求导 ($L \rightarrow a_2$)

现在，我们要将第一步和第二步结合。根据多元微积分的链式法则，损失 $\ell$ 对线性输出向量 $\mathbf{a}_2$ 的梯度，等于损失对概率向量的梯度 乘以 雅可比矩阵：
$$\frac{\partial \ell}{\partial \mathbf{a}_2} = \frac{\partial \ell}{\partial \mathbf{y}} \cdot J$$

展开为代数求和的形式，我们要计算 $\ell$ 对某一个具体线性输出 $a_s$ 的导数：
$$\frac{\partial \ell}{\partial a_s} = \sum_{j=0}^{9} \frac{\partial \ell}{\partial y_j} \cdot \frac{\partial y_j}{\partial a_s}$$

**魔法时刻 1：[雅可比矩阵的坍缩](./explains/06_jacobian_matrix_collapse.md)**
回想 6.4 节，除了 $j=t$ 的位置，$\frac{\partial \ell}{\partial y_j}$ 全都是 $0$！

这意味着，长达 10 项的求和公式直接坍缩，只剩下唯一的一项：
$$\frac{\partial \ell}{\partial a_s} = \frac{\partial \ell}{\partial y_t} \cdot \frac{\partial y_t}{\partial a_s} = \left(-\frac{1}{y_t}\right) \cdot \frac{\partial y_t}{\partial a_s}$$

**魔法时刻 2：复杂分式的完美抵消**
现在，把 6.5 节求得的雅可比矩阵元素代入进来：

* **如果 $s = t$（即求导位置恰好是正确类别的打分）：**
    代入情况 A 的结果：
    $$\frac{\partial \ell}{\partial a_t} = \left(-\frac{1}{y_t}\right) \cdot \left[ y_t(1 - y_t) \right] = -(1 - y_t) = \mathbf{y_t - 1}$$
* **如果 $s \neq t$（即求导位置是错误类别的打分）：**
    代入情况 B 的结果：
    $$\frac{\partial \ell}{\partial a_s} = \left(-\frac{1}{y_t}\right) \cdot \left[ -y_t y_s \right] = \mathbf{y_s} = \mathbf{y_s - 0} $$

**极简结论：**
无论是哪种情况，最终结果都可以统一为一句极度优美的话：**损失对该层线性输出的梯度，就等于网络的预测概率 减去 该类别的真实标签（1 或 0）。**

若推广到 $n$ 个样本求平均，用矩阵形式表达就是：
$$da_2 = \frac{1}{n}(y - T)$$
*(其中 $T$ 是正确位置为 1、其余为 0 的 One-Hot 标签矩阵)*

---

## 6.7 第三步：线性输出对参数及前一层求导 ($a_2 \rightarrow W_2, b_2, z_1$)

源头梯度 $da_2$ 拿到了，接下来的推导就是纯粹的矩阵乘法了。
第二层前向传播公式：$a_2 = z_1 \cdot W_2 + b_2$

1.  **对权重 $W_2$ 求导：**
    根据矩阵微积分法则，梯度等于输入端 $z_1$ 的转置 乘以 输出端梯度 $da_2$：
    $$dW_2 = z_1^T \cdot da_2$$
2.  **对偏置 $b_2$ 求导：**
    因为偏置是对所有 $n$ 个样本进行“广播”相加的，反向传播时需要把这 $n$ 个样本的梯度按列累加：
    $$db_2 = \sum_{i=1}^{n} da_2^{(i)}$$
3.  **向上一层回传梯度：**
    为了让网络继续反向传播，需要求出对隐藏层输出 $z_1$ 的梯度：
    $$dz_1 = da_2 \cdot W_2^T$$

---

## 6.8 第四步：梯度穿透隐藏层激活函数 ($z_1 \rightarrow a_1$)

梯度来到了第一层的 Sigmoid 激活函数：$z_1 = \sigma(a_1)$。

Sigmoid 的导数公式为：$\sigma'(x) = \sigma(x)(1 - \sigma(x))$。
由于激活函数是逐元素独立运算的（没有雅可比矩阵复杂的交叉项），我们可以直接将传回来的梯度 $dz_1$ 与局部导数进行**逐元素相乘**（记为 $\odot$）：
$$da_1 = dz_1 \odot \left( z_1 \odot (1 - z_1) \right)$$

---

## 6.9 第五步：第一层参数的梯度 ($a_1 \rightarrow W_1, b_1$)

这就完全是 6.7 节的重演了。
第一层前向公式：$a_1 = X \cdot W_1 + b_1$

权重梯度：
$$dW_1 = X^T \cdot da_1$$

偏置梯度（对 $n$ 个样本求和）：
$$db_1 = \sum_{i=1}^{n} da_1^{(i)}$$

---

## 6.10 反向传播：极其优雅的代码实现

将上述纯数学公式翻译成代码，利用 NumPy 的高级索引，我们连构造 One-Hot 矩阵的内存都省了：

```python
def _backprop_gradient(self, X, t):
    n = X.shape[0]
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    # ================= 前向传播 =================
    a1 = X @ W1 + b1               
    z1 = sigmoid(a1)               
    a2 = z1 @ W2 + b2              
    y  = softmax(a2)               

    # ================= 反向传播 =================
    # 6.4 - 6.6: Softmax 与交叉熵联合梯度 (抵消坍缩后的极简实现)
    da2 = y.copy()
    da2[np.arange(n), t] -= 1       # 精准在正确类别的位置减 1
    da2 /= n

    # 6.7: 第二层参数与回传梯度
    dW2 = z1.T @ da2                
    db2 = np.sum(da2, axis=0)       
    dz1 = da2 @ W2.T                

    # 6.8: 穿透 Sigmoid 激活函数
    da1 = dz1 * (z1 * (1 - z1))     # '*' 代表逐元素相乘

    # 6.9: 第一层参数梯度
    dW1 = X.T @ da1                
    db1 = np.sum(da1, axis=0)      

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
```

## 6.11 为什么不用数值梯度？

你可能会问：既然可以通过微小的变化 $h$ 来近似计算导数（数值梯度），为什么还要费劲推导反向传播？

$$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta_i + h) - L(\theta_i - h)}{2h}$$

* **数值梯度：** 要更新 39,760 个参数，你需要做近 **80,000 次**前向传播。慢如蜗牛，但代码好写，通常只用来**做测试，验证反向传播代码是否写错了**。
* **反向传播：** 只需要 **1 次**前向传播 + **1 次**反向计算。速度快约 1000 倍。这是工业界实际训练的唯一方案。

> **下一章预告**：现在我们拿到了所有参数的“指导意见”（梯度），接下来该怎么走？SGD、Momentum、Adam 等各种高级“走法”即将登场。

---

[← 第 5 章](chapter-05-loss-functions.md) | [返回目录](README.md) | [第 7 章：优化器 →](chapter-07-optimizers.md)

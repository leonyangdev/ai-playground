这是一个非常关键的误解！你正好卡在了整个推导中最容易把人绕晕的 **“下标”** 上。

你之所以觉得这里应该是 0，是因为你把 $\frac{\partial \ell}{\partial y_j}$ 和 $\frac{\partial \ell}{\partial y_t}$ 搞混了。让我们放慢动作，把这层窗户纸捅破。

### 1. 谁是 0，谁不是 0？

回想一下我们的交叉熵损失函数：$\ell = -\log(y_t)$。（$t$ 是正确类别的索引）。

如果网络预测了 3 个类别的概率：$y_0, y_1, y_2$，并且真实的正确类别是 $t=1$。
那么损失仅仅是：$\ell = -\log(y_1)$。

现在，我们拿这个 $\ell$ 去对所有的 $y$ 求偏导：
* 对 $y_0$ 求导：公式里没有 $y_0$，所以 $\frac{\partial \ell}{\partial y_0} = \mathbf{0}$
* **对 $y_1$（也就是 $y_t$）求导**：根据对数求导法则，$\frac{\partial \ell}{\partial y_1} = \mathbf{-\frac{1}{y_1}}$ **（注意！这个绝对不是 0）**
* 对 $y_2$ 求导：公式里没有 $y_2$，所以 $\frac{\partial \ell}{\partial y_2} = \mathbf{0}$

**结论：** 只有当对**错误类别**的概率求导时，结果才是 0。对**正确类别 $t$** 的概率求导时，结果是 $-\frac{1}{y_t}$。

### 2. 求和公式是如何坍缩的？

根据链式法则，我们要求 $\ell$ 对某个错误类别分数 $a_s$ 的导数：
$$\frac{\partial \ell}{\partial a_s} = \sum_{j} \frac{\partial \ell}{\partial y_j} \cdot \frac{\partial y_j}{\partial a_s}$$

展开来看（假设有 3 个类别，正确是 $t=1$，我们正在对错误的 $a_0$ 也就是 $s=0$ 求导）：
$$= \left( \frac{\partial \ell}{\partial y_0} \cdot \frac{\partial y_0}{\partial a_0} \right) + \left( \frac{\partial \ell}{\partial y_1} \cdot \frac{\partial y_1}{\partial a_0} \right) + \left( \frac{\partial \ell}{\partial y_2} \cdot \frac{\partial y_2}{\partial a_0} \right)$$

现在，把第一步求出的“0”和“非0”代入进去：
$$= \left( \mathbf{0} \cdot \dots \right) + \left( \mathbf{-\frac{1}{y_1}} \cdot \frac{\partial y_1}{\partial a_0} \right) + \left( \mathbf{0} \cdot \dots \right)$$

看到了吗？**正是因为那些 0 的存在，把周围没用的项全部“杀”掉了！** 整个长长的求和公式，最后只活下来了中间那一项（也就是 $j = t$ 的那一项）。

### 3. 活下来的那一项是什么？

活下来的这一项是：
$$\frac{\partial \ell}{\partial y_t} \cdot \frac{\partial y_t}{\partial a_s}$$

* **左半边 $\frac{\partial \ell}{\partial y_t}$：** 它是 $-\frac{1}{y_t}$，**不是 0**。
* **右半边 $\frac{\partial y_t}{\partial a_s}$：** 这是在问：“错误类别的打分 $a_s$ 变化时，正确类别的概率 $y_t$ 会怎么变？”因为 $t \neq s$（一个是正确类别，一个是错误类别，下标不同），所以应用我们在上一节讲的“非对角线（挤压）”法则，结果是 $-y_t y_s$。

把它们乘在一起：
$$\left(-\frac{1}{y_t}\right) \cdot \left[ -y_t y_s \right]$$

负负得正，分母的 $y_t$ 和分子的 $y_t$ 约掉，最后就得出了完美的结果：$y_s$。

***

**总结一下你的困惑点：**
0 乘以任何数确实是 0，但在链式法则的求和公式中，**乘以 0 的那些项（$j \neq t$）已经被彻底消灭丢弃了**。我们公式里剩下的 $\left(-\frac{1}{y_t}\right)$，是茫茫 0 海中**唯一一个不是 0 的幸存者**。
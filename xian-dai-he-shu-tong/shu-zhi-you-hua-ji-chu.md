# 数值优化基础

## 一、数值稳定性

1. 在计算机中执行数学运算需要使用有限的比特位来表达实数，这会引入近似误差。

   **近似误差可以在多步数值运算中传递、积累，从而导致理论上成功的算法失败。因此数值算法设计时要考虑将累计误差最小化。**

2. 当从头开始实现一个数值算法时，需要考虑数值稳定性。当使用现有的数值计算库（如`tensorflow` ）时，不需要考虑数值稳定性。

#### 1.1 上溢出、下溢出

1. 一种严重的误差是下溢出`underflow`：当接近零的数字四舍五入为零时，发生下溢出。

   许多函数在参数为零和参数为一个非常小的正数时，行为是不同的。如：对数函数要求自变量大于零，除法中要求除数非零。

2. 一种严重的误差是上溢出`overflow`：当数值非常大，超过了计算机的表示范围时，发生上溢出。
3. 一个数值稳定性的例子是`softmax`函数。

   设 $$\mathbf{\vec x}=(x_1,x_2,\cdots,x_n)^{T}$$ ，则`softmax`函数定义为： $$\text{softmax}(\mathbf{\vec x})=\left(\frac{\exp(x_1)}{\sum_{j=1}^{n}\exp(x_j)},\frac{\exp(x_2)}{\sum_{j=1}^{n}\exp(x_j)},\cdots,\frac{\exp(x_n)}{\sum_{j=1}^{n}\exp(x_j)}\right)^{T}$$ 

   当所有的 $$x_i$$ 都等于常数 $$c$$ 时，`softmax`函数的每个分量的理论值都为 $$\frac 1n$$ 。

   * 考虑 $$c$$ 是一个非常大的负数（比如趋近负无穷），此时 $$\exp( c)$$ 下溢出。此时 $$ \frac{\exp(c )}{\sum_{j=1}^{n}\exp(c )}$$  分母为零，结果未定义。
   * 考虑 $$c$$ 是一个非常大的正数（比如趋近正无穷），此时 $$\exp( c)$$  上溢出。  的结果未定义。

4. 为了解决`softmax`函数的数值稳定性问题，令 $$\mathbf{\vec z}=\mathbf{\vec x}-\max_i x_i$$ ，则有 $$\text{softmax}(\mathbf{\vec z}) $$ 的第 $$i$$  个分量为： $$\text{softmax}(\mathbf{\vec z})_i=\frac{\exp(z_i)}{\sum_{j=1}^{n}\exp(z_j)}=\frac{\exp(\max_k x_k)\exp(z_i)}{\exp(\max_k x_k)\sum_{j=1}^{n}\exp(z_j)}\\ =\frac{\exp(z_i+\max_k x_k)}{\sum_{j=1}^{n}\exp(z_j+\max_k x_k)}\\ =\frac{\exp(x_i)}{\sum_{j=1}^{n}\exp(x_j)}\\ =\text{softmax}(\mathbf{\vec x})_i$$ 
   * 当 $$\mathbf{\vec x} $$ 的分量较小时， $$\mathbf{\vec z} $$  的分量至少有一个为零，从而导致 $$\text{softmax}(\mathbf{\vec z})_i$$ 的分母至少有一项为 1，从而解决了下溢出的问题。
   * 当 $$\mathbf{\vec x} $$ 的分量较大时， $$\text{softmax}(\mathbf{\vec z})_i$$ 相当于分子分母同时除以一个非常大的数 $$\exp(\max_i x_i)$$ ，从而解决了上溢出。
5. 当 $$\mathbf{\vec x} $$  的分量 $$x_i$$ 较小时，  $$\text{softmax}(\mathbf{\vec x})_i$$ 的计算结果可能为 0 。此时 $$\log \text{softmax}(\mathbf{\vec x})$$  趋向于负无穷，因此存在数值稳定性问题。
   * 通常需要设计专门的函数来计算 $$\log \text{softmax}$$，而不是将 $$\text{softmax}$$ 的结果传递给 $$\text{log}$$ 函数。
   * $$\log\text{softmax}(\cdot)$$函数应用非常广泛。通常将 $$\text{softmax}$$ 函数的输出作为模型的输出。由于一般使用样本的交叉熵作为目标函数，因此需要用到 $$\text{softmax}$$ 输出的对数。
6. `softmax` 名字的来源是`hardmax`。
   * `hardmax` 把一个向量 $$\mathbf{\vec x} $$  映射成向量 $$(0,\cdots,0,1,0,\cdots,0)^T$$  。即： $$\mathbf{\vec x} $$ 最大元素的位置填充`1`，其它位置填充`0`。
   * `softmax` 会在这些位置填充`0.0~1.0` 之间的值（如：某个概率值）。

#### 1.2 Conditioning

1. `Conditioning`刻画了一个函数的如下特性：当函数的输入发生了微小的变化时，函数的输出的变化有多大。

   对于`Conditioning`较大的函数，在数值计算中可能有问题。因为函数输入的舍入误差可能导致函数输出的较大变化。

2. 对于方阵 $$\mathbf A\in \mathbb R^{n\times n}$$ ，其条件数`condition number`为： $$\text{condition number}=\max_{1\le i,j\le n,i\ne j}\left|\frac{\lambda_i}{\lambda_j} \right|$$ 

   其中 $$\lambda_i,i=1,2,\cdots,n$$ 为 $$A$$ 的特征值。

   * **方阵的条件数就是最大的特征值除以最小的特征值。**
   * **当方阵的条件数很大时，矩阵的求逆将对误差特别敏感（即：  的一个很小的扰动，将导致其逆矩阵一个非常明显的变化）。**
   * **条件数是矩阵本身的特性，它会放大那些包含矩阵求逆运算过程中的误差。**




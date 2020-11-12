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

## **二、梯度下降法**

1. 梯度下降法是求解无约束最优化问题的一种常见方法，优点是实现简单。
2. 对于函数： ****$$f:\mathbb R^{n} \rightarrow \mathbb R$$ ，假设输入 $$\mathbf{\vec x}=(x_1,x_2,\cdots,x_n)^{T}$$ ，则定义梯度： $$\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x})=\left(\frac{\partial}{\partial x_1}f(\mathbf{\vec x}),\frac{\partial}{\partial x_2}f(\mathbf{\vec x}),\cdots,\frac{\partial}{\partial x_n}f(\mathbf{\vec x})\right)^{T}$$ 

   函数的驻点满足： $$\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x})=\mathbf{\vec 0}$$ 。

3. 沿着方向 $$\mathbf{\vec u}$$ 的方向导数`directional derivative`定义为： $$\lim_{\alpha\rightarrow 0}\frac{f(\mathbf{\vec x}+\alpha\mathbf{\vec u})-f(\mathbf{\vec x})}{\alpha} $$ 

   其中 $$\mathbf{\vec u}$$ 为单位向量。

   方向导数就是 $$\frac{\partial}{\partial \alpha}f(\mathbf{\vec x}+\alpha\mathbf{\vec u})$$ 。根据链式法则，它也等于 $$\mathbf{\vec u}^{T}\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x})$$ 。

4. 为了最小化 $$f$$ ，则寻找一个方向：沿着该方向，函数值减少的速度最快（换句话说，就是增加最慢）。即： $$\min_{\mathbf{\vec u}}  \mathbf{\vec u}^{T}\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x})\\ s.t.\quad ||\mathbf{\vec u}||_2=1$$ 

   假设 $$\mathbf{\vec u}$$ 与梯度的夹角为 $$\theta$$ ，则目标函数等于： $$||\mathbf{\vec u}||_2||\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x})||_2 \cos\theta$$ 。

   考虑到 $$||\mathbf{\vec u}||_2=1$$ ，以及梯度的大小与 $$\theta$$ 无关，于是上述问题转化为： $$\min_\theta \cos\theta$$ 

   于是： $$\theta^{*}=\pi$$ ，即 $$\mathbf{\vec u}$$ 沿着梯度的相反的方向。即：**梯度的方向是函数值增加最快的方向，梯度的相反方向是函数值减小的最快的方向。** [**该问题**](https://zhuanlan.zhihu.com/p/38525412) **（**需要注意的是，若 切线的斜率大于 ![\[&#x516C;&#x5F0F;\]](https://www.zhihu.com/equation?tex=0) 时，表明函数在该点有上升趋势；当其小于 ![\[&#x516C;&#x5F0F;\]](https://www.zhihu.com/equation?tex=0) 时，表明函数在该点有下降趋势。$$cos\pi = -1，f<0$$即在该点有下降的趋势，若$$\theta = 0$$，方向导数与梯度（多元偏导数）一致，大于0，是增加的趋势**）**

   因此：可以沿着负梯度的方向来降低 $$f$$ 的值，这就是梯度下降法。

5. 根据梯度下降法，为了寻找 $$f$$ 的最小点，迭代过程为： $$\mathbf{\vec x}^{\prime}= \mathbf{\vec x}-\epsilon\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x})$$ 。其中： $$\epsilon$$ 为学习率，它是一个正数，决定了迭代的步长。

   迭代结束条件为：梯度向量 $$\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x})$$ 的每个成分为零或者非常接近零。

6. 选择学习率有多种方法：
   * 一种方法是：选择 $$\epsilon$$ 为一个小的、正的常数。
   * 另一种方法是：给定多个 $$\epsilon$$ ，然后选择使得 $$f(\mathbf{\vec x}-\epsilon\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x}))$$ 最小的那个值作为本次迭代的学习率（即：选择一个使得目标函数下降最大的学习率）。

     这种做法叫做线性搜索`line search` 。

   * 第三种方法是：求得使 $$f(\mathbf{\vec x}-\epsilon\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x}))$$ 取极小值的 $$\epsilon$$ ，即求解最优化问题： $$\epsilon^{*}=\arg\min_{\epsilon,\epsilon \gt 0 }f(\mathbf{\vec x}-\epsilon\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x}))$$ 

     这种方法也称作最速下降法。

     * 在最速下降法中，假设相邻的三个迭代点分别为： $$\mathbf{\vec x}^{<k>},\mathbf{\vec x}^{<k+1>},\mathbf{\vec x}^{<k+2>}$$ ，可以证明： $$(\mathbf{\vec x}^{<k+1>}-\mathbf{\vec x}^{<k>})\cdot (\mathbf{\vec x}^{<k+2>}-\mathbf{\vec x}^{<k+1>})=0$$ 。即相邻的两次搜索的方向是正交的！

       证明： $$\mathbf{\vec x}^{<k+1>}=\mathbf{\vec x}^{<k>}-\epsilon^{<k>}\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x}^{<k>})\\ \mathbf{\vec x}^{<k+2>}=\mathbf{\vec x}^{<k+1>}-\epsilon^{<k+1>}\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x}^{<k+1>})$$ 

       根据最优化问题，有： $$\epsilon^{<k>}=\arg\min_{\epsilon,\epsilon \gt 0 }f(\mathbf{\vec x}^{<k+1>})$$ 

       将 $$\mathbf{\vec x}^{<k+1>}=\mathbf{\vec x}^{<k>}-\epsilon\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x}^{<k>})$$ 代入，有： $$f(\mathbf{\vec x}^{<k+1>})=f(\mathbf{\vec x}^{<k>}-\epsilon\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x}^{<k>}))$$ 

       为求 $$f(\mathbf{\vec x}^{<k+1>})$$ 极小值，则求解： $$ \frac{\partial f(\mathbf{\vec x}^{<k>}-\epsilon\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x}^{<k>})) }{\partial \epsilon}\mid_{\epsilon=\epsilon^{<k>}}=0$$  。

       根据链式法则： $$\frac{\partial f(\mathbf{\vec x}^{<k>}-\epsilon\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x}^{<k>})) }{\partial \epsilon}= \nabla _{\mathbf{\vec x}} f(\mathbf{\vec x}^{<k>}-\epsilon\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x}^{<k>}))\cdot[- \nabla _{\mathbf{\vec x}} f(\mathbf{\vec x}^{<k>})] = 0$$ 

       即： $$ \nabla _{\mathbf{\vec x}} f(\mathbf{\vec x}^{<k+1>})\cdot \nabla _{\mathbf{\vec x}} f(\mathbf{\vec x}^{<k>})=0$$ 。则有： $$(\mathbf{\vec x}^{<k+2>}-\mathbf{\vec x}^{<k+1>})\cdot (\mathbf{\vec x}^{<k+1>}-\mathbf{\vec x}^{<k>})  =0$$ 。

     * 此时迭代的路线是锯齿形的，因此收敛速度较慢。
7. 某些情况下如果梯度向量 $$\nabla _{\mathbf{\vec x}} f(\mathbf{\vec x})$$ 的形式比较简单，则可以直接求解方程： 。

   此时不用任何迭代，直接获得解析解。

8. 梯度下降算法：
   * 输入：  
     * 目标函数 $$f(\mathbf {\vec x})$$ 
     * 梯度函数 $$g(\mathbf {\vec x})=\nabla f(\mathbf {\vec x}) $$ 
     * 计算精度 $$e$$ 
   * 输出： $$f(\mathbf {\vec x})$$ 的极小点 $$\mathbf {\vec x}^*$$ 
   * 算法步骤：
     * 选取初始值 $$\mathbf {\vec x}^{<0>}\in \mathbb R^{n}$$ ，置  $$k = 0$$ 。
     * 迭代，停止条件为：梯度收敛或者目标函数收敛。迭代步骤为：
       * 计算目标函数 $$f(\mathbf {\vec x}^{<k>})$$ ，计算梯度  $$\mathbf {\vec g}_k=g(\mathbf {\vec x}^{<k>})$$ 。
       * 若梯度 $$|\mathbf {\vec g}_k| \lt e$$ ，则停止迭代， $$\mathbf {\vec x}^*=\mathbf {\vec x}$$ 。
       * 若梯度 $$|\mathbf {\vec g}_k| \ge e$$ ，则令 $$\mathbf {\vec p}_k=-\mathbf {\vec g}_k$$ ，求 $$\epsilon_k$$ ： $$\epsilon_k =\min_{\epsilon  \le 0}f(\mathbf {\vec x}^{<k>}+\epsilon  \mathbf {\vec p}_k)$$  。

         > 通常这也是个最小化问题。但是可以给定一系列的 $$\epsilon_k$$ 的值，如：`[10,1,0.1,0.01,0.001,0.0001]` 。然后从中挑选使得目标函数最小的那个。

       * 令 $$\mathbf {\vec x}^{<k+1>} = \mathbf {\vec x}^{<k>}+\epsilon_k \mathbf {\vec p}_k$$ ，计算  $$f(\mathbf {\vec x}^{<k+1>})$$ 。
         * 若 $$|f(\mathbf {\vec x}^{<k+1>})-f(\mathbf {\vec x}^{<k>})| \lt e$$ 或者 $$|\mathbf {\vec x}^{<k+1>}-\mathbf {\vec x}^{<k>}| \lt e$$ 时，停止迭代，此时 $$\mathbf {\vec x}^*=\mathbf {\vec x}$$ 。
         * 否则，令  $$k=k+1$$ ，计算梯度 $$\mathbf {\vec g}_k=g(\mathbf {\vec x}^{<k>})$$  继续迭代。
9. 当目标函数是凸函数时，梯度下降法的解是全局最优的。通常情况下，梯度下降法的解不保证是全局最优的。
10. 梯度下降法的收敛速度未必是最快的。

![](../.gitbook/assets/image%20%2831%29.png)

> 以下是最小二乘的线性回归

```text
class MyLinearRegression:
    def __init__(self, X, Y):
        self.X = np.column_stack((np.array([1]*np.array(X).shape[0]).reshape(-1,1),np.array(X)))
        self.Y = np.array(Y).reshape(-1,1)

    def GradientDescent(self,alpha,n_rounds):
        n_feature = self.X.shape[1]
        Beta = np.array([0.0]*n_feature).reshape(-1,1)
        for i in range(n_rounds):
            #计算epsilon
            epsilon = self.Y - np.dot(self.X,Beta)
            if np.sum(epsilon*epsilon) < 1e-4:
                break
            for j in range(n_feature):
                Beta[j] = Beta[j] - alpha*np.dot(epsilon.T,self.X[:,j])
        return Beta
```



\*\*\*\*

\*\*\*\*




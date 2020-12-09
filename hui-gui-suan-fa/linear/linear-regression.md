---
description: Linear Regression
---

# 线性回归

### 前言

回归分析是基于观测数据建立变量间适当的依赖关系，以分析数据的内在规律，并用于预报和控制等问题。是研究变量间相关关系的一种统计方法。

$$
Y = f(x_1,x_2,...,x_k)+\epsilon
$$

其中 $$f$$ 表示: $$Y$$ 是自变量 $$x_1,x_2,..x_k$$ 的函数，是 $$Y$$ 关于 $$X$$ 的回归函数。 $$\epsilon$$ 是由其他多种因素综合的效果是一个随机变量，即随机误差。因为希望 $$f(x_1,x_2,...,x_k)$$ 的值要在Y值中起主要的作用，故要求 $$E(\epsilon)=0,D(\epsilon)=\sigma^2>0$$ 尽量小。

### 相关定义和假设

**在实际应用中,回归函数** $$f$$ **选用的是最简单的线性函数，称之为线性模型，另外在实际应用中，给定了自变量值后，认为** $$Y$$ **服从正态分布。**

故线性回归模型可写为**:**

$$
Y = \beta_0+\beta_1X_1+\beta_2X_2+...+\beta_pX_p+\epsilon
$$

或写为**：**

$$
Y \sim N(\beta_0+\beta_1X_1+\beta_2X_2+...+\beta_pX_p,\sigma^2)
$$

因为观测值是一个向量而不是标量，故引入矩阵的表示是有必要的。为了方便，引入：

$$
Y = \left[ \begin{aligned} y_1 \\ y_2 \\ \vdots \\ y_n \end{aligned} \right], X = \left[ \begin{aligned} 1 && x_{11} && \cdots && x_{1p} \\ 1 && x_{21} && \cdots && x_{2p} \\ \vdots && \vdots && \ddots && \vdots \\ 1 && x_{n1} && \cdots && x_{np}  \end{aligned} \right],\beta = \left[ \begin{aligned} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_p \end{aligned} \right],\epsilon = \left[ \begin{aligned} \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n \end{aligned} \right]
$$

记为： ****$$Y = X\beta+\epsilon$$ 。

它的**基本假设**为：

* $$X$$ 是满秩矩阵，列满秩，且 $$x_1,x_2,...,x_n$$ 是确定变量不是随机变量， $$r(X) = p+1$$ 表示 $$x_1,x_2,...,x_p$$ 无多重共线性；  

> 即列满秩，列向量线性无关，即变量线性无关。对线性回归模型基本假设之一是自变量之间不存在严格的线性关系。如不然，则会对回归参数估计带来严重影响

> 复共线性（multi collinearity），一译“多重共线性”。 在回归分析中当自变量的个数很多，且相互之间相关很高时，由样本资料估计的回归系数的精度显著下降的现象。即自变量之间存在近似的线性关系。存在复共线性时最小二乘估计的精度下降。通常解决的办法：（1）通过变量选择使回归模型中的自变量减少，（2）作主成分回归分析（用少数几个主成分作为回归自变量）。

* $$E(\epsilon_i) = 0; cov(\epsilon_i,\epsilon_j)=\left\{ \begin{aligned} \sigma^2 && i=j \\ 0 && i \ne j\end{aligned}\right. \Rightarrow \epsilon \sim N(0,\sigma^2I_n)$$ 
*  $$y_1,y_2,...,y_n$$ 是相互独立的随机变量； $$Y \sim N(X\beta,\sigma^2I_n)$$ 

**求出的用于拟合散点图的直线方程可写为** $$\hat Y = \hat \beta_0 + \hat\beta_1X_1+\hat\beta_2X_2+...+\hat\beta_kX_k$$ **即** $$\hat Y= X \hat \beta$$ **为经验回归直线方程,** $$\hat \beta$$ **为** $$\beta$$ **的估计值。**

### 损失函数

为了求出 $$\beta$$ 的估计值，可以采用最小二乘法，梯度下降法，最大似然法等

**直观的想法，即是要求各样本点到经验回归直线的距离越小越好（最小二乘思想）**，损失函数可选为均方误差，平均绝对值误差，平均绝对百分比误差

均方误差 Mean Squared Error \(L2损失函数\)：

$$
MSE = \sum_{i=1}^n(y_i-\hat y_i)^2/n=  \sum_{i=1}^n\epsilon_i^2 /n
$$

平均绝对值误差 Mean Absolute Error \(L1损失函数\)：

$$
MAE = \sum_{i=1}^n|y_i-\hat y_i|/n = \sum_{i=1}^n\epsilon_i /n
$$

平均绝对百分比误差 Mean Absolute Percentage Error：

$$
MAPE = \sum_{i=1}^n|\frac{y_i-\hat y_i}{y_i}|/n
$$

> L1损失对异常值更加稳健，但其导数并不连续，因此求解效率很低。L2损失对异常值敏感，但给出了更稳定的闭式解（closed form solution）（通过将其导数设置为0）由于MSE对误差（e）进行平方操作（y - y\_predicted = e），如果e&gt; 1，误差的值会增加很多。如果我们的数据中有一个离群点，e的值将会很高，将会远远大于\|e\|。这将使得和以MAE为损失的模型相比，以MSE为损失的模型会赋予更高的权重给离群点。进而调整以最小化这个离群数据点，但是却是以牺牲其他正常数据点的预测效果为代价，这最终会降低模型的整体性能。
>
> MAE损失适用于训练数据被离群点损坏的时候（即，在训练数据而非测试数据中，我们错误地获得了不切实际的过大正值或负值）

### 参数求解

#### 最小二乘法求解 （最小二乘法一般以最小化数据集的均方误差MSE为目标）

**解析法求解**：即 $$\mathop {argmin}_{\hat \beta} MSE$$ 求偏导，令偏导数为0，即可求到极值点时的参数估计值

$$
\begin{aligned} \frac{\partial\theta}{\partial\beta_0} = \frac{\partial(-2\sum_{i=1}^n(y_i-(\hat\beta_0+\hat\beta_1x_{i1}+\hat\beta_2x_{i2}+...+\hat\beta_px_{ip}))/n}{\partial\beta_0} \\ \Rightarrow \overline{Y} = \hat\beta_0+\overline{X_1}\hat\beta_1+ \overline{X_2}\hat\beta_2 + ... + \overline{X_p}\hat\beta_p \\ \frac{\partial\theta}{\partial\beta_1} = \frac{\partial(-2\sum_{i=1}^n(y_i-(\hat\beta_0+\hat\beta_1x_{i1}+\hat\beta_2x_{i2}+...+\hat\beta_px_{ip}))x_{i1}/n}{\partial\beta_1}  \\ \Rightarrow \frac{1}{n}\sum_{i=1}^ny_ix_{i1}  = \overline{X_1}\hat\beta_0 +\frac{1}{n}(\sum_{i=1}^nx_{i1}^2\hat\beta_1 + \sum_{i=1}^nx_{i1}x_{i2}\hat\beta_2+...+\sum_{i=1}^nx_{i1}x_{ip}\hat\beta_p) \\ ....................................................................................\\\frac{1}{n}\sum_{i=1}^ny_ix_{ip}  = \overline{X_p}\hat\beta_0 +\frac{1}{n}(\sum_{i=1}^nx_{i1}x_{ip}\hat\beta_1 + \sum_{i=1}^nx_{ip}x_{i2}\hat\beta_2+...+\sum_{i=1}^nx_{ip}^2\hat\beta_p)\end{aligned}
$$

进而利用克莱姆法则求解方程组，这样求解比较麻烦，因而会采用矩阵形式 等价于求解残差平方和的最小值，故

$$
\begin{aligned}
\frac{\partial\theta}{\partial\hat\beta} = \frac{\partial(Y-X\hat\beta)^2}{\partial\hat\beta}\\\frac{\partial\theta}{\partial\hat\beta} = \frac{\partial(Y-X\hat\beta)^T(Y-X\hat\beta)}{\partial\hat\beta} \\= \frac{\partial(Y^T-\hat\beta^TX^T)(Y-X\hat\beta)}{\partial\hat\beta} \\ = \frac{\partial(Y^TY-\hat\beta^TX^TY-Y^TX\hat\beta+\hat\beta^TX^TX\hat\beta)}{\partial\hat\beta}
\end{aligned}
$$

因为 $$X^TX$$ 为对称阵，所以 $$\frac{\partial \hat\beta^TX^TX\hat\beta}{\partial \hat\beta} = \frac{\partial(X\hat\beta)^2}{\partial \hat\beta} = 2X^TX\hat\beta$$ 证明略；

同时上式 $$\hat\beta^TX^TY = \sum_{i=1}^ny_i(\hat\beta_0+\sum_{j=1}^px_{ij}\hat\beta_j) = Y^TX\hat\beta$$ 

$$
\frac{\partial\theta}{\partial\hat\beta} = -2X^TY+2X^TX\hat\beta = 0
$$

$$
X^TX\hat\beta = X^TY \Rightarrow \hat\beta = (X^TX)^{-1}X^TY
$$

因为假定了矩阵X是列满秩，故 $$X^TX$$ 可逆。

> 如果不是满秩， $$X^TX$$ 的行列式为0，不存在其逆矩阵，因而无法求出参数的估计值，故而线性回归需要假定无多重共线性

另一种推导方式：

$$
\begin{aligned} \frac{\partial\theta}{\partial\beta_0} \Rightarrow \sum_{i=1}^n\epsilon_i = 0 \Rightarrow \epsilon^T1 = 0 \\  \frac{\partial\theta}{\partial\beta_1} \Rightarrow  \sum_{i=1}^n\epsilon_ix_{i1} = \epsilon^TX_1 = 0 \\ .........................................\\ \frac{\partial\theta}{\partial\beta_p} \Rightarrow  \sum_{i=1}^n\epsilon_ix_{ip} = \epsilon^TX_p = 0\end{aligned}
$$

故既是求:

$$
\epsilon^T[1,X_1,X_2,...,X_p] = 0
$$

$$
\begin{aligned}
\epsilon^TX = 0 \Rightarrow (Y-X\hat\beta)^TX = 0\\X^T(Y-X\hat\beta) = 0 \Rightarrow X^TX\hat\beta = X^TY \\ \hat\beta = (X^TX)^{-1}X^TY
\end{aligned}
$$

**现实中** $$X^TX$$ **往往不是满秩矩阵，例如大量变量，其数目超过样例数，导致X的列数多于行数，** $$X^TX$$ **显然不满秩，此时可解出多个** $$\hat \beta$$ **,他们都能使均方误差最小，选择哪一个解作为输出，将由学习算法的归纳偏好决定。鉴于非满秩的矩阵在求逆时会出现问题， 一般会通过缩减系数来进行处理，采用的方法是引入正则项。**

**正则化向原始模型引入额外信息，也能够防止过拟合和提高模型泛化性能 （训练数据不够多时，或者overtraining时，常常会导致过拟合（overfitting））**  


------

正则化项：

1. L1正则化 **Lasso Regression**： 等价于求 $$(Y-X\hat\beta)^T(Y-X\hat\beta)+\lambda ||\hat\beta||_1$$ 最小化; $$\lambda > 0$$ 为正则系数，调整正则化项与训练误差的比例。  
2. L2正则化 **Ridge Regression**：等价于求 $$(Y-X\hat\beta)^T(Y-X\hat\beta)+\lambda ||\hat\beta||_2^2$$ 最小化; $$\lambda > 0$$ 为正则系数，调整正则化项与训练误差的比例。  最终系数 $$\hat \beta = (X^TX+\lambda I)^{-1}X^TY$$ 
3. 同时包含L1和L2的正则化 **Elastic Net**：等价于求 $$(Y-X\hat\beta)^T(Y-X\hat\beta)+\lambda \rho||\hat\beta||_1 + \frac{\lambda(1-\rho)}{2} ||\hat\beta||_2^2$$ 最小化; $$\lambda > 0$$ 为正则系数，调整正则化项与训练误差的比例。 $$1 \ge \rho \ge 0 $$ 为比例系数，调整L1正则化与L2正则化的比例。

作用：  
1. L1正则化可以产生**稀疏权值矩阵**，即产生一个稀疏模型，可以用于特征选择 2. L2正则化可以防止模型过拟合（overfitting）；一定程度上，L1也可以防止过拟合

[正则化，过拟合的解释](https://hit-scir.gitbooks.io/neural-networks-and-deep-learning-zh_cn/content/chap3/c3s5ss1.html)

> 拟合过程中通常都倾向于让权值尽可能小，最后构造一个所有参数都比较小的模型。因为一般认为参数值小的模型比较简单，能适应不同的数据集，也在一定程度上避免了过拟合现象。可以设想一下对于一个线性回归方程，若参数很大，那么只要数据偏移一点点，就会对结果造成很大的影响；但如果参数足够小，数据偏移得多一点也不会对结果造成什么影响，专业一点的说法是『抗扰动能力强』。

性质的区别总结如下：

| L2正则 | L1正则 |
| :---: | :---: |
| 计算效率高（因为有解析解） | 在非稀疏情形下计算效率低 |
| 非稀疏输出 | 稀疏输出 |
| 无特征选择 | 内置特征选择 |

内置特征选择是L1范数被经常提及的有用的性质，而L2范数并不具备。这是L1范数的自然结果，它趋向于产生稀疏的系数。假设模型有100个系数，但是仅仅只有其中的10个是非零的，这实际上是说“其余的90个系数在预测目标值时都是无用的”。L2范数产生非稀疏的系数，因此它不具备这个性质。

稀疏性指的是一个矩阵（或向量）中只有少数的项是非零的。L1范数具备性质：产生许多0或非常小的系数和少量大的系数。

计算效率。L1范数没有一个解析解，但是L2范数有。这就允许L2范数在计算上能高效地计算。然而，L1范数的解具备稀疏性，这就允许它可以使用稀疏算法，以使得计算更加高效。

------

#### 数值法求解：梯度下降优化求解（得到解析表达式很容易，使得导数为0 ，这就变成了一个方程式求解问题，计算机并不擅长，而且非常复杂，计算量大，且不好计算，而且会出现无解；而使用梯度下降法，就非常简单，而且计算机是非常擅长优化的；）

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

**由于回归模型中并不一定要求都是正态分布，如果不是正态随机变量，用最小二乘比较方便。**

#### 最大似然求解：在实际应用中，一般假定Y是正态随机变量，此时可用最大似然法求解。即在数据集给定的情况下，找出使该数据集发生的可能性最大的一组参数。

求解步骤：\(当似然函数L可微且参数集合是开集条件下\(L可微且似然方程有唯一解\)\) 

1. 由总体分布写出似然函数 $$L(\theta) = f(x_1)*f(x_2)...f(x_n) = \prod_{i=1}^nf(x_i)$$ 

2. 对对数似然函数关于参数求偏导，令导数等于0 

3. 解似然方程

$$
Y \sim N(\beta_0+\beta_1X_1+\beta_2X_2+...+\beta_pX_p,\sigma^2)
$$

故线性回归的似然方程：

$$
U = \prod_{i=1}^nf(y_i) = \prod_{i=1}^n\frac{1}{2\pi\sigma^2}^{\frac{1}{2}}e^{-\frac{1}{2\sigma^2}(y_i-\hat y_i)^2} = \frac{1}{2\pi\sigma^2}^{\frac{n}{2}} e^{-\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i-\hat y_i)^2}
$$

$$
lnL = -\frac{n}{2}ln(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i-\hat y_i)^2
$$

故要求得 $$lnL$$ 的最大值，只需求 $$\sum_{i=1}^n(y_i-\hat y_i)^2$$ 的最小值。**因而在正态分布的条件下，最小二乘法与最大似然估计法的结果是一致的**

**故而线性回归以均方误差作为优化目标，好处在于： 均方误差方便求解（求导） 均方误差有清晰的统计学解释（正态分布下的最大似然）**

\*\*\*\*

![](../../.gitbook/assets/image%20%2836%29.png)

![](../../.gitbook/assets/image%20%2837%29.png)










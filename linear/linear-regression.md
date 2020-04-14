---
description: 线性回归
---

# Linear Regression

#### 相关定义和假设

回归分析是基于观测数据建立变量间适当的依赖关系，以分析数据的内在规律，并用于预报和控制等问题。是研究变量间相关关系的一种统计方法。

$$
Y = f(x_1,x_2,...,x_k)+\epsilon
$$

其中 $$f$$ 表示: $$Y$$ 是自变量 $$x_1,x_2,..x_k$$ 的函数，是 $$Y$$ 关于 $$X$$ 的回归函数。 $$\epsilon$$ 是由其他多种因素综合的效果是一个随机变量，即随机误差。因为希望 $$f(x_1,x_2,...,x_k)$$ 的值要在Y值中起主要的作用，故要求 $$E(\epsilon)=0,D(\epsilon)=\sigma^2>0$$ 尽量小。

**在实际应用中,回归函数** $$f$$ **选用的是最简单的线性函数，称之为线性模型，另外在实际应用中，给定了自变量值后，认为** $$Y$$ **服从正态分布。**

故线性回归模型可写为**:**

$$
Y = \beta_0+\beta_1X_1+\beta_2X_2+...+\beta_pX_p+\epsilon
$$

基本假设： $$\epsilon \sim N(0,\sigma^2)$$ ；且 $$y_1,y_2,...,y_n$$ 是相互独立的随机变量； $$\epsilon_i$$ 也相互独立。 \(**是对Y的假设\)。**

或写为**：**

$$
Y \sim N(\beta_0+\beta_1X_1+\beta_2X_2+...+\beta_pX_p,\sigma^2)
$$

因为观测值是一个向量而不是标量，故引入矩阵的表示是有必要的。为了方便，引入：

$$
Y = \left[ \begin{aligned} y_1 \\ y_2 \\ \vdots \\ y_n \end{aligned} \right], X = \left[ \begin{aligned} 1 && x_{11} && \cdots && x_{1p} \\ 1 && x_{21} && \cdots && x_{2p} \\ \vdots && \vdots && \ddots && \vdots \\ 1 && x_{n1} && \cdots && x_{np}  \end{aligned} \right],\beta = \left[ \begin{aligned} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_p \end{aligned} \right],\epsilon = \left[ \begin{aligned} \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n \end{aligned} \right]
$$

记为： ****$$Y = X\beta+\epsilon$$ 。

它的基本假设为：

1. $$X$$ 是满秩矩阵，列满秩，且 $$x_1,x_2,...,x_n$$ 是确定变量不是随机变量， $$r(X) = p+1$$ 表示 $$x_1,x_2,...,x_p$$ 无多重共线性；  

> 即列满秩，列向量线性无关，即变量线性无关。对线性回归模型基本假设之一是自变量之间不存在严格的线性关系。如不然，则会对回归参数估计带来严重影响

> 复共线性（multi collinearity），一译“多重共线性”。 在回归分析中当自变量的个数很多，且相互之间相关很高时，由样本资料估计的回归系数的精度显著下降的现象。即自变量之间存在近似的线性关系。存在复共线性时最小二乘估计的精度下降。通常解决的办法：（1）通过变量选择使回归模型中的自变量减少，（2）作主成分回归分析（用少数几个主成分作为回归自变量）。


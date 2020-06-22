# 高斯混合聚类

1. 高斯混合聚类采用概率模型来表达聚类原型。
2. 对于 $$n$$ 维样本空间  $$\mathcal X$$ 中的随机向量 $$\mathbf{\vec x}$$ ，若 $$\mathbf{\vec x}$$ 服从高斯分布，则其概率密度函数为 ： $$p(\mathbf{\vec x}\mid \vec \mu,\Sigma)=\frac {1}{(2\pi)^{n/2}|\Sigma|^{1 /2}}\exp\left(-\frac 12(\mathbf{\vec x}- \vec \mu)^{T}\Sigma^{-1}(\mathbf{\vec x}- \vec \mu)\right)$$ 

   其中 $$\vec \mu=(\mu_{1},\mu_2,\cdots,\mu_n)^{T}$$ 为 $$n$$ 维均值向量， $$\Sigma$$  是 $$n\times n$$  的协方差矩阵。 $$\mathbf{\vec x}$$ 的概率密度函数由参数 $$\vec \mu,\Sigma$$ 决定。

3. 定义高斯混合分布： $$p_{\mathcal M}=\sum_{k=1}^{K}\alpha_k p(\mathbf{\vec x}\mid \vec \mu_k,\Sigma_k)$$  。该分布由 $$K$$个混合成分组成，每个混合成分对应一个高斯分布。其中:
   * $$\vec \mu_k,\Sigma_k$$ 是第 $$k$$个高斯混合成分的参数。
   * $$\alpha_k \gt 0$$ 是相应的混合系数，满足 $$\sum_{k=1}^{K}\alpha_k=1$$ 。
4. 假设训练集 $$\mathbb D=\{\mathbf{\vec x}_1,\mathbf{\vec x}_2,\cdots,\mathbf{\vec x}_N\}$$ 的生成过程是由高斯混合分布给出。

   令随机变量 $$Z \in \{1,2,\cdots,K\}$$ 表示生成样本 $$\mathbf{\vec x}$$ 的高斯混合成分序号，  $$Z$$ 的先验概率 $$P(Z =k)=\alpha_k$$ 。

   生成样本的过程分为两步：

   * 首先根据概率分布 $$\alpha_1,\alpha_2,\cdots,\alpha_K$$ 生成随机变量  $$Z$$ 。
   * 再根据 $$Z$$的结果，比如 $$Z=k$$ ， 根据概率 $$p(\mathbf{\vec x}\mid \vec \mu_k,\Sigma_k)$$ 生成样本。

5. 根据贝叶斯定理， 若已知输出为 $$\mathbf{\vec x}_i$$ ，则 $$Z$$ 的后验分布为： $$p_{\mathcal M}(Z =k\mid \mathbf{\vec x}_i)=\frac{P(Z =k)p_{\mathcal M}(\mathbf{\vec x}_i \mid Z =k)}{p_{\mathcal M}(\mathbf{\vec x}_i)} = \frac{\alpha_k  p(\mathbf{\vec x}_i\mid \vec \mu_k,\Sigma_k)}{\sum_{l=1}^{K}\alpha_l p(\mathbf{\vec x}_i\mid \vec \mu_l,\Sigma_l)}$$ 

   其物理意义为：所有导致输出为 $$\mathbf{\vec x}_i$$ 的情况中，  $$Z=k$$ 发生的概率。

6. 当高斯混合分布已知时，高斯混合聚类将样本集 $$\mathbb D$$ 划分成  $$K$$ 个簇 $$\mathcal C=\{\mathbb C_1,\mathbb C_2,\cdots,\mathbb C_K\}$$ 。

   对于每个样本 $$\mathbf{\vec x}_i$$ ，给出它的簇标记 $$\lambda_i$$ 为： $$\lambda_i=\arg\max_k p_{\mathcal M}(Z =k\mid \mathbf{\vec x}_i)$$ 

   即：如果 $$\mathbf{\vec x}_i$$ 最有可能是 $$Z =k$$ 产生的，则将该样本划归到簇 。

   这就是通过最大后验概率确定样本所属的聚类。

7. 现在的问题是，如何学习高斯混合分布的参数。由于涉及到隐变量  $$Z$$ ，可以采用`EM`算法求解。

   具体求解参考`EM` 算法的章节部分。


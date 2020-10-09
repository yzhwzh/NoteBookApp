---
description: Latent Dirichlet Allocation
---

# 潜在狄利克雷分配\(LDA\)

潜在狄利克雷分配\(LDA\)，作为基于贝叶斯学习的话题模型，是潜在语义分析、概率潜在语义分析的扩展，于2002年由Blei等提出。LDA在文本数据挖掘、图像处理、生物信息处理等领域被广泛使用。

LDA模型是文本集合的生成概率模型。假设每个文本由话题的一个多项式分布表示，每个话题由单词的一个多项式分布表示，**特别假设文本的话题分布的先验分布是狄利克雷分布，话题的单词分布的先验分布也是狄利克雷分布。先验分布的导入使LDA能够更好地应对话题模型学习的过拟合现象。**

LDA的文本集合的生成过程如下：首先随机生成一个文本话题分布，之后再该文本的每个位置，依据该文本的话题分布随机生成一个话题，然后在该位置依据该话题的单词分布随机生成一个单词，直至文本的最后一个位置，生成整个文本。重复以上的过程生成所有文本。

**LDA模型是含隐变量的概率图模型。模型中，每个话题的单词分布，每个文本的话题分布，文本的每个位置的话题是隐变量；**文本的每个文职的单词是观测变量。LDA模型的学习与推理无法直接求解，通常使用吉布斯抽样和变分EM算法。前者是蒙特卡洛法，后者是近似计算。

## 狄利克雷分布

#### 分布定义

**多项分布**是一种多元离散随机变量的概率分布，是二项分布的扩展。假设重复进行$$n$$次独立随机试验，每次试验可能出现的结果有$$k$$种，第$$i$$种结果出现的概率为![p\_i](https://math.jianshu.com/math?formula=p_i)，第i种结果出现的次数为![n\_i](https://math.jianshu.com/math?formula=n_i)。如果用随机变量![\mathbf {X=\{X\_1,X\_2,\dots,X\_k\}}](https://math.jianshu.com/math?formula=%5Cmathbf%20%7BX%3D%5C%7BX_1%2CX_2%2C%5Cdots%2CX_k%5C%7D%7D)，表示试验所有可能结果的次数，其中![\mathbf X\_i](https://math.jianshu.com/math?formula=%5Cmathbf%20X_i)表示第$$i$$种结果出现的次数，那么随机变量![\mathbf X](https://math.jianshu.com/math?formula=%5Cmathbf%20X)服从多项分布。 若多元离散随机变量的概率密度为

$$\begin{aligned} P(X_1=n_1,X_2=n_2,\dots,X_k=n_k)=\frac{n!}{n_1!n_2!\dots n_k!}p_1^{n_1}p_2^{n_2}\dots p_k^{n_k} \\ =\frac{n!}{\prod_{i=1}^k n_i! } \prod_{k=1}^k p_i^{n_i} \end{aligned}$$ 

 其中 $$p=(p_1,p_2,\dots,p_k),p_i \ge 0,i=1,2,\dots,k,\sum_{i=1}^k p_i = 1, \sum_{i=1}^k n_i = n$$ ,则称随机变量$$X$$服从参数为$$(n,p)$$的多项分布，记作![X \sim Mult\(n,p\)](https://math.jianshu.com/math?formula=X%20%5Csim%20Mult%28n%2Cp%29)

**狄利克雷分布**是一种多元随机变量的概率分布，是贝塔分布的扩展。在贝爷斯学习中，狄利克雷分布作为多项分布的先验概率使用（ 贝塔分布（Beta Distribution\) 是一个作为伯努利分布和二项式分布的共轭先验分布的密度函数。在概率论中，**贝塔分布**，也称**Β分布，**是指一组定义在\(0,1\) 区间的连续概率分布。）。

 多元**连续型**随机变量![\theta = \(\theta\_1,\theta\_2,\dots,\theta\_k\)](https://math.jianshu.com/math?formula=%5Ctheta%20%3D%20%28%5Ctheta_1%2C%5Ctheta_2%2C%5Cdots%2C%5Ctheta_k%29)的概率密度函数为

$$p(\theta|\alpha)=\frac{\Gamma(\sum_{i=1}^k \alpha_i)}{\prod_{i=1}^k \Gamma(\alpha_i)}\prod_{i=1}^k \theta_i^{\alpha_i-1}$$ 

 其中 $$\sum_{i=1}^k \Theta_i = 1,\Theta_i \ge 0,\alpha=(\alpha_1,\alpha_2,\dots,\alpha_k) ,\alpha_i > 0,i=1,2,\dots,k$$ ，称随机变量![\Theta](https://math.jianshu.com/math?formula=%5CTheta)服从参数为![\alpha](https://math.jianshu.com/math?formula=%5Calpha)的狄利克雷分布，记作![\Theta \sim Dir\(\alpha\)](https://math.jianshu.com/math?formula=%5CTheta%20%5Csim%20Dir%28%5Calpha%29), 式中：$$\Gamma(s) = \int_{0}^{\infty} x^{s-1}e^{-x} ds,s > 0$$ 具有以下性质 $$\Gamma(s+1) = s\Gamma(s)$$ 当$$s$$是自然数时，有 $$\Gamma(s+1) = s!$$ ,令 $$B(a) = \frac{\prod_{i=1}^k \Gamma(a_i)}{\Gamma (\sum_{i=1}^k a_i)}$$ ，则狄利克雷分布的密度函数可以写成 $$p(\Theta|a) = \frac{1}{B(a)}\prod_{k=1}^k \Theta^{a_i-1}$$ ； ![B\(a\)](https://math.jianshu.com/math?formula=B%28a%29)是规范化因子，称为多元贝塔函数\(称为扩展的贝塔函数\)。由密度函数性质 $$\int \frac{\Gamma(\sum_{i=1}^k a_i)}{\prod_{i=1}^k \Gamma(a_i)}\prod_{i=1}^k\Theta^{a_i-1} d\Theta =\frac{\Gamma(\sum_{i=1}^k a_i)}{\prod_{i=1}^k \Gamma(a_i)} \int \prod_{i=1}^k\Theta^{a_i-1} d\Theta =1$$ 得 $$B(a) = \int \prod_{i=1}^k \Theta^{a_i-1} d\Theta$$ 。

狄利克雷有一些重要性质:\(1\)狄利克雷分布属于指数分布簇\(2\)狄利克雷分布是多项分布的共轭先验。贝叶斯学习中常使用共轭分布，如果后验分布与先验分布属于同类，则先验分布与后验分布称为共轭分布，先验分布称为共轭先验。**如果多项分布的先验分布是狄利克雷分布，作为先验分布的狄利克雷分布的参数又称为超参数，使用共轭先验分布的好处是便于从先验分布计算后验分布。**

 将样本数据表示为D，目标是计算样本数据D给定条件下参数![\Theta](https://math.jianshu.com/math?formula=%5CTheta)的后验概率![p\(\Theta\|D\)](https://math.jianshu.com/math?formula=p%28%5CTheta%7CD%29)，对于给定样本数据D，似然函数是 $$p(D|\theta)=\theta_1^{n_1}\theta_2^{n_2}\dots\theta_k^{n_k}=\prod_{i=1}^k\theta_i^{n_i}$$ ； 假设随机变量![\theta](https://math.jianshu.com/math?formula=%5Ctheta)服从狄利克雷分布![p\(\theta\|a\)](https://math.jianshu.com/math?formula=p%28%5Ctheta%7Ca%29)其中![a=\(a\_1,a\_2,\dots,a\_k\)](https://math.jianshu.com/math?formula=a%3D%28a_1%2Ca_2%2C%5Cdots%2Ca_k%29)为参数，则![\theta](https://math.jianshu.com/math?formula=%5Ctheta)的先验分布为 $$p(\theta|a)=\frac{\Gamma(\sum_{i=1}^k a_i)}{\prod_{i=1}^k \Gamma(a_i)}\prod_{i=1}^k \theta^{a_i-1}=\frac{1}{B(a)}\prod_{i=1}^k \theta_i^{a_i-1} = Dir(\theta|a),a>0$$ 

 根据贝爷斯规则，在给定样本数据D和参数a的条件下，![\theta](https://math.jianshu.com/math?formula=%5Ctheta)的后验概率分布是 $$\begin{aligned} p(\theta|D,a) = \frac{p(D|\theta)p(\theta|a)}{p(D|\alpha)} \\ =\frac{\prod_{i=1}^k \theta_i^{n_i}\frac{1}{B(a)}\theta_i^{a_i-1}}{\int \prod_{i=1}^k \theta_i^{n_i}\frac{1}{B(a)}\theta_i^{a_i-1}d\theta} \\ =\frac{1}{B(a+n)}\prod_{i=1}^k \theta_i^{a_i+n_i+1} \\ =Dir(\theta|a+n) \end{aligned}$$ 

 狄利克雷的后验分布等于狄利克雷分布参数![a=\(a\_1,a\_2,\dots,a\_k\)](https://math.jianshu.com/math?formula=a%3D%28a_1%2Ca_2%2C%5Cdots%2Ca_k%29)加上多项分布的观测技术![n=\(n\_1,n\_2,\dots,n\_k\)](https://math.jianshu.com/math?formula=n%3D%28n_1%2Cn_2%2C%5Cdots%2Cn_k%29)

备注：

![](../.gitbook/assets/image%20%2830%29.png)




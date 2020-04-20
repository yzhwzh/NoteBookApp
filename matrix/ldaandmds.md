---
description: Linear Discriminant Analysis  and Multidimensional Scaling
---

# 线性判别分析与多维尺度变换

PCA，SVD，LDA，MDS求解近似矩阵，都涉及特征值，特征向量求解，矩阵相似变换。

### LDA \(线性判别分析\)

        LDA是与PCA非常相关的一种算法，PCA是将变量投影到方差最大的基向量上，而LDA则加入了类别标签，投影后希望组内尽可能近，组间尽可能远，因而LDA属于监督学习的一种降维算法。

**推导过程**：

        假设找到最佳的一个投影基向量 $$W$$  \(列向量\)，那原始数据 $$X$$  在 $$W$$ 上的投影可以表示为 $$W^T X$$ 。设类标签为 $$w$$  ，均值向量为 $$\mu_i=\frac{1}{N_i}\sum_{x_i\in \omega_i}x_i$$  ，方差为 $$\sigma =\sum_{x_i\in \omega_i} (x_i-\mu_i)(x_i-\mu_i)^T$$ 

投影之后的均值向量为：

$$
\widetilde \mu_i = \frac{1}{N_i}\sum_{x_i\in \omega_i}W^Tx_i = W^T\frac{1}{N_i}\sum_{x_i\in \omega_i}x_i = W^T \mu_i
$$

**首先考虑二分类情况**

        假如只有两个类别，什么是最佳的 $$W$$  呢？首先发现，能够使投影后的两类样本中心点尽量分离的直线是好的直线，定量表示就是： $$||{\mu_0} - {\mu_1}||_2^2$$ ；可用 $$\underset{w}{argmax}J(w) = ||\overset{\sim}{\mu_0} - \overset{\sim}{\mu_1}||_2^2$$  作为目标函数。

但是只考虑 $$J(w)$$  是不行的，因为投影到基向量上，虽然可能获得最大的中心点间距，但类别之间可能发生重叠，如下图：

![](https://pic3.zhimg.com/80/v2-94af3d4648725a854b5456d0d8f253f2_720w.jpg)

因而还需要考虑同类样本点之间的方差，同类样本点之间方差越小， 就越难以分离。故而引入另一个度量值，散列值 $$\overset{\sim}{S_i^2} = \sum_{x_i\in \omega_i}(W^Tx _i- \overset{\sim}{\mu_i})^2$$  即投影后的组内方差。因想要的投影后不同类别的样本点越分开越好，同类的越聚集越好，也就是均值差越大越好，散列值越小越好。故目标函数可以变为 

$$
\underset{w}{argmax}J(w) = \frac{||\overset {\sim}{\mu_0}- \overset{\sim}{\mu_1}||_2^2}{\overset{\sim}{S_0^2}+\overset{\sim}{S_1^2}}
$$

 ，寻找 $$W$$  使该函数最大化 ；展开散列公式， 

$$
\overset{\sim}{S_i^2} = \sum_{x_i\in \omega_i}(W^Tx_i - \overset{\sim}{\mu_i})^2 =\sum_{x_i\in \omega_i} (W^Tx_i - W^T \mu_i)(W^Tx_i - W^T \mu_i)^T=\sum_{x_i\in \omega_i} W^T(x_i-\mu_i)(x_i-\mu_i)^TW
$$

故分母 $$\overset{\sim}{S_0^2}+\overset{\sim}{S_1^2}$$  可简化成 $$W^TS_wW$$  ; 其中 $$S_w = \sigma_0+\sigma_1$$  定义为类内散度矩阵；

分子

$$
||\overset {\sim}{\mu_0}- \overset{\sim}{\mu_1}||_2^2 = (W^T\mu_0-W^T\mu_1)(W^T\mu_0-W^T\mu_1)^T = W^T(\mu_0-\mu_1)(\mu_0-\mu_1)^TW = W^TS_bW
$$

  ; $$S_b$$  为类间散度矩阵；

因而目标函数 $$J(w) = \frac{w^TS_bw}{w^TS_ww}$$ 

在我们求导之前，需要对分母进行归一化，因为不做归一的话， $$W$$  扩大任何倍，都成立，我们就无法确定 $$W$$  。因此我们打算令 $$||W^TS_wW|| = 1$$  ，即引入新的约束条件，那么加入拉格朗日乘子后进行求导。

引入拉格朗日乘子： 目标函数等价于 

$$c(w) = -W^TS_bW + \lambda (W^TS_wW-1) \Rightarrow \frac{dc}{dW} = -2 S_bW+ 2\lambda S_wW = 0 \Rightarrow S_bW = \lambda S_wW$$ 

如果 $$S_w$$  可逆，则 $$S_w^{-1}S_bW = \lambda W$$  即可知 $$W$$  为 $$S_w^{-1}S_b$$  的特征向量。

注意到 $$S_bW = (\mu_0-\mu_1)(\mu_0-\mu_1)^TW$$  ，两个类别下 $$(\mu_0-\mu_1)^TW$$  为常数，不妨令 $$S_bW=\lambda_w (\mu_0-\mu_1)$$；带入上式求得 $$\lambda W=S_w^{-1}(\mu_0-\mu_1)\lambda_w$$；由于对 $$W$$  扩大缩小任何倍不影响结果，因此可以约去两边的未知常数 $$\lambda,\lambda_w$$  ，得到 $$W=S_w^{-1}(\mu_0-\mu_1)$$  也就是说我们只要求出原始二类样本的均值和方差就可以确定最佳的投影方向了。

**多类别情况**

        假设有C个类别，需要K维向量（或者叫做基向量）来做投影。

同理，K维基向量作为行向量组成的矩阵， 即 $$W^T$$  可以用来进行投影变换, 即 $$y=W^Tx$$ 

同样从类内散度矩阵和类间散度矩阵来考虑，类内散度矩阵不变： $$S_{wi} = \sum_{x\in \omega_i} (x-\mu_i)(x-\mu_i)^T;S_w = \sum_{i=1}^CS_{wi}$$  仍可以理解为类内部样本点的方差。

类间散度矩阵：二分类的时候，度量的是两类样本点的距离；多类的时候，度量的是每类均值点相对于样本中心的散列情况，

![](https://pic4.zhimg.com/80/v2-d7b31eb53764cafb4d4232545c64288b_720w.jpg)

即 $$S_B=\sum_{i=1}^CN_i(\mu_i-\mu)(\mu_i-\mu)^T$$  此处引入 $$N_i$$  ,是考虑到某类样本点较多，则可赋予更多的权重，理应用 $$N_i/N$$  表示，但由于J\(w\)对倍数不敏感，因此使用$$N_i$$  ；其中 $$\mu=\frac{1}{N}\sum_{\forall x} x = \frac{1}{N}\sum_{x_i\in \omega_i}N_iu_i$$ 

投影后的变换：

$$
\overset{\sim}{S_w} = \sum_{i=1}^C\sum_{y_i\in \omega_i}(y_i-\overset{\sim}{u_i})(y_i-\overset{\sim}{u_i})^T=W^TS_wW
$$

$$
\overset{\sim}{S_B}=\sum_{i=1}^CN_i(\tilde{\mu_i}-\tilde{\mu})(\tilde{\mu_i}-\tilde{\mu})^T = W^TS_BW
$$

故最终 $$J(w) = \frac{\tilde{S_W}}{\tilde{S_B}}$$ 

同样引入拉格朗日乘子，限定 $$||W^TS_BW||$$  为1；求得 $$S_W^{-1}S_BW = \lambda W$$  即可知 $$W$$  为 $$S_W^{-1}S_B$$  的特征向量;取前K个特征向量组成W矩阵即可；

由于 $$S_B$$  中 $$(\mu_i-\mu)$$  秩为1，因此$$S_B$$  的秩至多为C（矩阵的秩小于等于各个相加矩阵的秩的和）。由于知道了前C-1个 $$\mu_i$$  后，最后一个 $$\mu_c$$  可以有前面的$$\mu_i$$   来线性表示，因此$$S_B$$ 的秩至多为C-1。那么K最大为C-1，即特征向量最多有C-1个。特征值大的对应的特征向量分割性能最好。

由于 $$S_W^{-1}S_B$$  不一定是对称阵，因此得到的K个特征向量不一定正交，这也是与PCA不同的地方。

### **MDS \(多维尺度变换\)**

        将高维坐标中的点投影到低维空间中，保持点彼此之间的相似性尽可能不变。相似性用距离来表示。其中采用欧式距离的称为Classical MDS\(经典多维尺度变换\)，否则为No-classical MDS\(非经典多维度尺度变换\)

**推导**：

        假设有 $$m$$  个样本，其样本空间如下： $$T = {x_1,x_2,x_3,...,x_m},x_i\in R^d$$  ,令 $$D \in R^{m*m}$$ 表示样本间的距离，其中第 $$i$$ 行 $$j$$ 列的元素 $$dist_{ij}$$ 表示样本 $$x_i$$ 与样本 $$x_j$$ 之间的距离。在不改变样本间距离的情况下，实现数据降维，最终要得到新的样本空间 $$Z \in R^{n*m},n \lt d$$ ; 此时要满足

$$
d_{ij}^2 = ||z_i-z_j||^2=||z_i||^2+||z_j||^2-2z_i^Tz_j
$$

条件； 由 $$z_i^Tz_j$$ 可以联想到内积矩阵，因而令 $$B=Z^TZ$$ ;保持维度仍为 $$m*m$$，故

$$
d_{ij}^2 = b_{ii}+b_{jj} - 2b_{ij}
$$

；在Z维空间中空间中，点可以进行平移与旋转，因此在Z维空间中会有多种分布满足要求，不失一般性，假设Z维空间中的实例点是中心化的， $$\sum_{i=1}^mz_i = 0$$ ；即B的行和列之和均为0， $$\sum_{i=1}^m b_{ij} = \sum_{j=1}^m b_{ij} = 0$$；进一步能得到

$$
\sum_{i=1}^m d_{ij}^2 = tr(B) + m b_{jj} \Rightarrow b_{jj} = \frac{1}{m}(\sum_{i=1}^m d_{ij}^2 - tr(B))
$$

$$
\sum_{j=1}^m d_{ij}^2 = tr(B) + m b_{ii} \Rightarrow b_{ii} = \frac{1}{m}(\sum_{j=1}^m d_{ij}^2 - tr(B))
$$

$$
\sum_{i=1}^m\sum_{j=1}^m d_{ij}^2 = 2mtr(B) \Rightarrow tr(B) = \frac{1}{2m} \sum_{i=1}^m\sum_{j=1}^m d_{ij}^2
$$

此时，行均值

$$
d_{i·}^2 =  \frac{1}{m} \sum_{j=1}^m d_{ij}^2 \Rightarrow b_{ii} = d_{i·}^2 - \frac{1}{2m^2} \sum_{i=1}^m\sum_{j=1}^m d_{ij}^2
$$

列均值

$$
d_{·j}^2 =  \frac{1}{m} \sum_{i=1}^m d_{ij}^2 \Rightarrow b_{jj} = d_{·j}^2 - \frac{1}{2m^2} \sum_{i=1}^m\sum_{j=1}^m d_{ij}^2
$$

均值

$$
d_{··}^2 =  \frac{1}{m^2} \sum_{i=1}^m \sum_{j=1}^m d_{ij}^2
$$

最终得到：

$$
b_{ij} = -\frac{1}{2}(d_{ij}^2-d_{i·}^2 + \frac{1}{2}d_{··}^2 -d_{·j}^2 + \frac{1}{2}d_{··}^2 ) = -\frac{1}{2}(d_{ij}^2-d_{i·}^2 -d_{·j}^2 + d_{··}^2)
$$

因而此时能求出内积矩阵B； 因为 $$B=Z^TZ$$ ,为对称阵，故存在正交矩阵 $$V$$ ，使得 $$B = V\Lambda V^T$$ 因而得出 $$Z = \sqrt{\Lambda} V^T$$ , 因为 $$Z = \sqrt{\Lambda} V^T$$ ，为了有效降维，此时可取 $$n \ll m$$ 个特征值，以及对应的特征向量来近似。


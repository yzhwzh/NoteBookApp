# 聚类算法

## 介绍

1. 在无监督学习`(unsupervised learning)`中，训练样本的标记信息是未知的。
2. 无监督学习的目标：通过对无标记训练样本的学习来揭露数据的内在性质以及规律。
3. 一个经典的无监督学习任务：寻找数据的最佳表达`(representation)`。常见的有：
   * 低维表达：试图将数据（位于高维空间）中的信息尽可能压缩在一个较低维空间中。
   * 稀疏表达：将数据嵌入到大多数项为零的一个表达中。该策略通常需要进行维度扩张。
   * 独立表达：使数据的各个特征相互独立。
4. 无监督学习应用最广的是聚类`(clustering)` 。
   * 给定数据集 $$\mathbb D=\{\mathbf {\vec x}_1,\mathbf {\vec x}_2,\cdots,\mathbf {\vec x}_N\}$$ ，聚类试图将数据集中的样本划分为 $$K$$ 个不相交的子集 $$\{\mathbb C_1,\mathbb C_2,\cdots,\mathbb C_K\}$$ ，每个子集称为一个簇`cluster`。其中： $$\mathbb C_k \bigcap_{k\ne l} \mathbb C_l=\phi,\quad \mathbb D=\bigcup _{k=1}^{K}\mathbb C_k$$ 。
   * 通过这样的划分，每个簇可能对应于一个潜在的概念。这些概念对于聚类算法而言，事先可能是未知的。
   * 聚类过程仅仅能自动形成簇结构，簇所对应的概念语义需要由使用者来提供。
5. 通常用 $$\lambda_i \in \{1,2,\cdots,K\}$$ 表示样本 $$\mathbf {\vec x}_i$$ 的簇标记`cluster label`，即 $$\mathbf {\vec x}_i \in \mathbb C_{\lambda_i}$$ 。于是数据集 $$\mathbb D$$ 的聚类结果可以用包含  个元素的簇标记向量 $$\vec \lambda=(\lambda_1,\lambda_2,\cdots,\lambda_N)^{T}$$ 来表示。
6. 聚类的作用：
   * 可以作为一个单独的过程，用于寻找数据内在的分布结构。
   * 也可以作为其他学习任务的前驱过程。如对数据先进行聚类，然后对每个簇单独训练模型。
7. 聚类问题本身是病态的。即：没有某个标准来衡量聚类的效果。
   * 可以简单的度量聚类的性质，如每个聚类的元素到该类中心点的平均距离。

     但是实际上不知道这个平均距离对应于真实世界的物理意义。

   * 可能很多不同的聚类都很好地对应了现实世界的某些属性，它们都是合理的。

     如：在图片识别中包含的图片有：红色卡车、红色汽车、灰色卡车、灰色汽车。可以聚类成：红色一类、灰色一类；也可以聚类成：卡车一类、汽车一类。

     > 解决该问题的一个做法是：利用深度学习来进行分布式表达，可以对每个车辆赋予两个属性：一个表示颜色、一个表示型号。

## 一、性能度量

1. 聚类的性能度量也称作聚类的有效性指标`validity index` 。
2. 直观上看，希望同一簇的样本尽可能彼此相似，不同簇的样本之间尽可能不同。即：簇内相似度`intra-cluster similarity`高，且簇间相似度`inter-cluster similarity`低。
3. 聚类的性能度量分两类：
   * 聚类结果与某个参考模型`reference model`进行比较，称作外部指标`external index` 。
   * 直接考察聚类结果而不利用任何参考模型，称作内部指标`internal index` 。

#### 1.1 外部指标

1. 对于数据集 $$\mathbb D=\{\mathbf {\vec x}_1,\mathbf {\vec x}_2,\cdots,\mathbf {\vec x}_N\}$$ ，假定通过聚类给出的簇划分为 $$\mathcal C=\{\mathbb C_1,\mathbb C_2,\cdots,\mathbb C_K\}$$ 。参考模型给出的簇划分为 $$\mathcal C^{*}=\{\mathbb C_1^{*},\mathbb C_2^{*},\cdots,\mathbb C_{K^\prime}^{*}\}$$ ，其中 $$K$$ 和 $$K^\prime$$ 不一定相等 。

   令 $$\vec \lambda,\vec \lambda^{*}$$ 分别表示 $$\mathcal C,\mathcal C^{*}$$  的簇标记向量。定义： $$ a=|SS|,SS=\{(\mathbf {\vec x}_i,\mathbf {\vec x}_j) \mid \lambda_i=\lambda_j,\lambda_i^{*}=\lambda_j^{*}, i \lt j\} \\    b=|SD|,SD=\{(\mathbf {\vec x}_i,\mathbf {\vec x}_j) \mid \lambda_i = \lambda_j,\lambda_i^{*} \ne \lambda_j^{*}, i \lt j\} \\    c=|DS|,DS=\{(\mathbf {\vec x}_i,\mathbf {\vec x}_j) \mid \lambda_i \ne \lambda_j,\lambda_i^{*}=\lambda_j^{*}, i \lt j\} \\    d=|DD|,DD=\{(\mathbf {\vec x}_i,\mathbf {\vec x}_j) \mid \lambda_i\ne\lambda_j,\lambda_i^{*}\ne\lambda_j^{*}, i \lt j\}$$ 

   其中 $$|\cdot|$$ 表示集合的元素的个数。各集合的意义为：

   * SS：包含了同时隶属于 $$\mathcal C,\mathcal C^{*}$$ 的样本对。
   * SD：包含了隶属于  $$C$$ ，但是不隶属于 $$\mathcal C^{*}$$  的样本对。
   * DS：包含了不隶属于 $$C$$ ， 但是隶属于 $$\mathcal C^{*}$$ 的样本对。
   * DD：包含了既不隶属于 $$C$$ ， 又不隶属于 $$\mathcal C^{*}$$  的样本对。

   由于每个样本对  $$(\mathbf {\vec x}_i,\mathbf {\vec x}_j), i\lt j$$ 仅能出现在一个集合中，因此有 $$a+b+c+d=\frac {N(N-1)}{2}$$ 。

2. 下述性能度量的结果都在 `[0,1]`之间。**这些值越大，说明聚类的性能越好。**

**1.1.1 Jaccard系数**

1. `Jaccard`系数`Jaccard Coefficient:JC`： $$JC=\frac {a}{a+b+c}$$ 

   它刻画了：所有的同类的样本对（要么在 $$\mathcal C$$ 中属于同类，要么在 $$\mathcal C^{*}$$ 中属于同类）中，同时隶属于  的样本对的比例。

**1.1.2 FM指数**

1. `FM`指数`Fowlkes and Mallows Index:FMI`： $$FMI=\sqrt{\frac {a}{a+b}\cdot \frac{a}{a+c}}$$ 

   它刻画的是：

   * 在 $$\mathcal C$$ 中同类的样本对中，同时隶属于 $$\mathcal C^{*}$$ 的样本对的比例为  $$p_1=\frac{a}{a+b}$$ 。
   * 在 $$\mathcal C^{*}$$ 中同类的样本对中，同时隶属于 $$\mathcal C$$ 的样本对的比例为  $$p_2=\frac{a}{a+c}$$ 。
   * `FMI`就是 $$p_1$$ 和 $$p_2$$ 的几何平均。

**1.1.3 Rand指数**

1. `Rand`指数`Rand Index:RI`： $$RI=\frac{a+d}{N(N-1)/2}$$ 

   它刻画的是：

   * 同时隶属于 $$\mathcal C,\mathcal C^{*}$$ 的同类样本对（这种样本对属于同一个簇的概率最大）与既不隶属于 $$C$$ 、 又不隶属于 $$\mathcal C^{*}$$ 的非同类样本对（这种样本对不是同一个簇的概率最大）之和，占所有样本对的比例。
   * 这个比例其实就是聚类的可靠程度的度量。

**1.1.4 ARI指数**

1. 使用`RI`有个问题：对于随机聚类，`RI`指数不保证接近0（可能还很大）。

   `ARI`指数就通过利用随机聚类来解决这个问题。

2. 定义一致性矩阵为： $$\begin{array} {c|crc}   & \mathbb C_1^{*} & \mathbb C_2^{*}&\cdots&\mathbb C_{K^\prime}^{*}&\text{sums} \\ \mathbb C_1  & \hline n_{1,1} &  n_{1,2}  &\cdots & n_{1,K^\prime}& s_1 \\ \mathbb C_2&  n_{2,1} &  n_{2,2}  &\cdots & n_{2,K^\prime}& s_2\\ \vdots  &\vdots & \vdots & \ddots&\vdots&\vdots\\ \mathbb  C_K&  n_{K,1} &  n_{K,2}  &\cdots & n_{K,K^\prime}& s_K\\ \text{sums}  & t_1& t_2 & \cdots&t_K&N \end{array}$$ 

   其中：

   *  $$s_i$$ 为属于簇 $$\mathbb C_i$$ 的样本的数量， $$t_i$$ 为属于簇 $$\mathbb C_i^{*}$$ 的样本的数量。
   *  $$n_{i,j}$$ 为同时属于簇 $$\mathbb C_i$$ 和簇 $$\mathbb C_i^{*}$$ 的样本的数量。

   则根据定义有： $$a=\sum_i\sum_jC_{n_{i,j}}^2$$  ，其中 $$C_{n}^2 = \frac{n(n-1)}{2}$$ **表示组合数。数字`2` 是因为需要提取两个样本组成样本对**。

3. 定义`ARI`指数`Adjusted Rand Index`: $$ARI=\frac{\sum_i\sum_jC^2_{n_{i,j}}-\left[\sum_iC^2_{s_i}\times \sum_jC^2_{t_j}\right]/C_N^2}{\frac 12\left[\sum_iC^2_{s_i}+\sum_jC^2_{t_j}\right]-\left[\sum_iC^2_{s_i}\times \sum_jC^2_{t_j}\right]/C_N^2}$$ 

   其中：

   * $$\sum_i\sum_jC^2_{n_{i,j}}$$ ：表示同时隶属于 $$\mathcal C,\mathcal C^{*}$$ 的样本对。
   *  $$\frac 12\left[\sum_iC^2_{s_i}+\sum_jC^2_{t_j}\right]$$ ：表示最大的样本对。

     即：无论如何聚类，同时隶属于 $$\mathcal C,\mathcal C^{*}$$ 的样本对不会超过该数值。

   *  $$\left[\sum_iC^2_{s_i}\times \sum_jC^2_{t_j}\right]/C_N^2$$ ：表示在随机划分的情况下，同时隶属于 $$\mathcal C,\mathcal C^{*}$$  的样本对的期望。
     * 随机挑选一对样本，一共有 $$C_N^2$$  种情形。
     * 这对样本隶属于 $$C$$ 中的同一个簇，一共有 $$\sum_iC^2_{s_i}$$  种可能。
     * 这对样本隶属于 $$\mathcal C^{*}$$ 中的同一个簇，一共有 $$\sum_jC^2_{t_i}$$ 种可能。
     * 这对样本隶属于 $$C$$ 中的同一个簇、且属于 $$\mathcal C^{*}$$ 中的同一个簇，一共有 $$\sum_iC^2_{s_i}\times \sum_jC^2_{t_j}$$ 种可能。
     * 则在随机划分的情况下，同时隶属于 $$\mathcal C,\mathcal C^{*}$$ 的样本对的期望（平均样本对）为： $$\left[\sum_iC^2_{s_i}\times \sum_jC^2_{t_j}\right]/C_N^2$$ 。

#### 1.2 内部指标

1. 对于数据集 $$\mathbb D=\{\mathbf {\vec x}_1,\mathbf {\vec x}_2,\cdots,\mathbf {\vec x}_N\}$$  ，假定通过聚类给出的簇划分为 $$\mathcal C=\{\mathbb C_1,\mathbb C_2,\cdots,\mathbb C_K\}$$ 。

   定义：

 $$\text{avg}(\mathbb C_k)=\frac{2}{|\mathbb C_k|(|\mathbb C_k|-1)}\sum_{\mathbf {\vec x}_i,\mathbf {\vec x}_j \in \mathbb C_k,i\ne j}\text{distance}(\mathbf {\vec x}_i,\mathbf {\vec x}_j),\quad k=1,2,\cdots,K$$

 $$\text{diam}(\mathbb C_k)=\max_{\mathbf {\vec x}_i,\mathbf {\vec x}_j \in \mathbb C_k,i\ne j}\text{distance}(\mathbf {\vec x}_i,\mathbf {\vec x}_j),\quad k=1,2,\cdots,K$$ 

$$d_{min}(\mathbb C_k,\mathbb C_l)=\min_{\mathbf {\vec x}_i \in \mathbb C_k,\mathbf {\vec x}_j \in \mathbb C_l}\text{distance}(\mathbf {\vec x}_i,\mathbf {\vec x}_j),\quad k,l=1,2,\cdots,K;k\ne l$$ 

$$d_{cen}(\mathbb C_k,\mathbb C_l)=\text{distance}( \vec \mu _k, \vec \mu _l),\quad k,l=1,2,\cdots,K;k\ne l$$ 

其中： $$\text{distance}(\mathbf {\vec x}_i,\mathbf {\vec x}_j)$$ 表示两点 $$\mathbf {\vec x}_i,\mathbf {\vec x}_j$$ 之间的距离； $$\vec \mu _k$$ 表示簇 $$\mathbb C_k$$  的中心点，  $$\vec \mu _l$$ 表示簇 $$\mathbb C_l$$ 的中心点， $$\text{distance}( \vec \mu _k, \vec \mu _l)$$ 表示簇 $$\mathbb C_k,\mathbb C_l$$ 的中心点之间的距离。

上述定义的意义为：

* $$\text{avg}(\mathbb C_k)$$ ： 簇 $$\mathbb C_k$$ 中每对样本之间的平均距离。
* $$\text{diam}(\mathbb C_k)$$ ：簇 $$\mathbb C_k$$ 中距离最远的两个点的距离。
* $$d_{min}(\mathbb C_k,\mathbb C_l)$$ ：簇 $$\mathbb C_k,\mathbb C_l$$ 之间最近的距离。
*  $$d_{cen}(\mathbb C_k,\mathbb C_l)$$ ：簇 $$\mathbb C_k,\mathbb C_l$$ 中心点之间的距离。

**1.2.1 DB指数**

1. `DB`指数`Davies-Bouldin Index:DBI`： $$DBI=\frac 1K \sum_{k=1}^{K}\max_{k\ne l}\left(\frac{\text{avg}(\mathbb C_k)+\text{avg}(\mathbb C_l)}{d_{cen}(\mathbb C_k,\mathbb C_l)}\right)$$ 

   其物理意义为：

   * 给定两个簇，每个簇样本距离均值之和比上两个簇的中心点之间的距离作为度量。

     该度量越小越好。

   * 给定一个簇 $$k$$ ，遍历其它的簇，寻找该度量的最大值。
   * 对所有的簇，取其最大度量的均值。

2. 显然 $$DBI$$ 越小越好。
   * 如果每个簇样本距离均值越小（即簇内样本距离都很近），则 $$DBI$$ 越小。
   * 如果簇间中心点的距离越大（即簇间样本距离相互都很远），则 $$DBI$$ 越小。

**1.2.2 Dunn指数**

1. `Dunn`指数`Dunn Index:DI`： $$DI=  \frac{\min_{k\ne l} d_{min}(\mathbb C_k,\mathbb C_l)}{\max_{i}\text{diam}(\mathbb C_i)}$$ 

   其物理意义为：任意两个簇之间最近的距离的最小值，除以任意一个簇内距离最远的两个点的距离的最大值。

2. 显然 $$DI$$ 越大越好。
   * 如果任意两个簇之间最近的距离的最小值越大（即簇间样本距离相互都很远），则 $$DI$$ 越大。
   * 如果任意一个簇内距离最远的两个点的距离的最大值越小（即簇内样本距离都很近），则 $$DI$$ 越大。

#### 1.3 距离度量

1. 距离函数 $$\text{distance}( \cdot, \cdot )$$ 常用的有以下距离：
   * 闵可夫斯基距离`Minkowski distance`：

     给定样本 $$\mathbf {\vec x}_i=(x_{i,1},x_{i,2},\cdots,x_{i,n})^{T},\mathbf {\vec x}_j=(x_{j,1},x_{j,2},\cdots,x_{j,n})^{T}$$ ，则闵可夫斯基距离定义为： $$\text{distance}( \mathbf {\vec x}_i, \mathbf {\vec x}_j )=\left(\sum_{d=1}^{n}|x_{i,d}-x_{j,d}|^{p}\right)^{1/p}$$ 

     * 当 $$p=2$$ 时，闵可夫斯基距离就是欧式距离`Euclidean distance`：
     * 当 $$p=1$$ 时，闵可夫斯基距离就是曼哈顿距离`Manhattan distance`：

   * `VDM`距离`Value Difference Metric`：

     考虑非数值类属性（如属性取值为：`中国，印度，美国，英国`），令 $$m_{d,a}$$ 表示 $$x_d=a$$  的样本数； $$m_{d,a,k}$$ 表示 $$x_d=a$$ 且位于簇 $$\mathbb C_k$$ 中的样本的数量。则在属性  $$d$$ 上的两个取值 $$VDM$$ 之间的 `VDM`距离为： $$VDM_p(a,b)=\left(\sum_{k=1}^{K}\left|\frac {m_{d,a,k}}{m_{d,a}}-\frac {m_{d,b,k}}{m_{d,b}}\right|^{p}\right)^{1/p}$$ 

     该距离刻画的是：属性取值在各簇上的频率分布之间的差异。
2. 当样本的属性为数值属性与非数值属性混合时，可以将闵可夫斯基距离与 `VDM` 距离混合使用。

   假设属性 $$x_1,x_2,\cdots,x_{n_c} $$ 为数值属性， 属性  $$x_{n_c+1},x_{n_c+2},\cdots,x_{n} $$ 为非数值属性。则： $$\text{distance}( \mathbf {\vec x}_i, \mathbf {\vec x}_j )=\left (\sum_{d=1}^{n_c}|x_{i,d}-x_{j,d}|^{p}+\sum_{d=n_c+1}^{n}VDM_p(x_{i,d},x_{j,d})^p\right)^{1/p}$$ 

3. 当样本空间中不同属性的重要性不同时，可以采用加权距离。

   以加权闵可夫斯基距离为例： $$\text{distance}( \mathbf {\vec x}_i, \mathbf {\vec x}_j )=\left(\sum_{d=1}^{n}w_d\times|x_{i,d}-x_{j,d}|^{p}\right)^{1/p}\\ w_d \ge 0,d=1,2,\cdots,n;\quad \sum_{d=1}^{n}w_d=1$$ 

4. 这里的距离函数都是事先定义好的。在有些实际任务中，有必要基于数据样本来确定合适的距离函数。这可以通过距离度量学习`distance metric learning`来实现。
5. 这里的距离度量满足三角不等式： $$\text{distance}( \mathbf {\vec x}_i, \mathbf {\vec x}_j ) \le \text{distance}( \mathbf {\vec x}_i, \mathbf {\vec x}_k )+\text{distance}( \mathbf {\vec x}_k, \mathbf {\vec x}_j )$$ 。

   在某些任务中，根据数据的特点可能需要放松这一性质。如：`美人鱼`和`人`距离很近，`美人鱼`和`鱼`距离很近，但是`人`和`鱼`的距离很远。这样的距离称作非度量距离`non-metric distance`。

### 二、原型聚类

1. 原型聚类`prototype-based clustering`假设聚类结构能通过一组原型刻画。

   常用的原型聚类有：

   * `k`均值算法`k-means` 。
   * 学习向量量化算法`Learning Vector Quantization:LVQ` 。
   * 高斯混合聚类`Mixture-of-Gaussian` 。


---
description: Feature Selection
---

# 特征选择

### 介绍

1. 对于一个学习任务，给定了属性集，其中某些属性可能对于学习来说是很关键的，但是有些属性可能就意义不大。

   * 对当前学习任务有用的属性称作相关特征`relevant feature` 。
   * 对当前学习任务没有用的属性称作无关特征`irrelevant feature` 。

   从给定的特征集合中选出相关特征子集的过程称作特征选择`feature selection`。

2. 特征选择可能会降低模型的预测能力。因为被剔除的特征中可能包含了有效的信息，抛弃了这部分信息会一定程度上降低预测准确率。

   这是计算复杂度和预测能力之间的折衷：

   * 如果保留尽可能多的特征，则模型的预测能力会有所提升，但是计算复杂度会上升。
   * 如果剔除尽可能多的特征，则模型的预测能力会有所下降，但是计算复杂度会下降。

### 特征选择原理

1. 特征选择是一个重要的数据预处理（`data preprocessing`）过程。在现实机器学习任务中，获取数据之后通常首先进行特征选择，然后再训练学习器。

   进行特征选择的原因：  


   * 首先，在现实任务中经常会遇到维数灾难问题，这是由于属性过多造成的。如果能从中选择出重要的特征，使得后续学习过程仅仅需要在一部分特征上构建模型，则维数灾难问题会大大减轻。

     从这个意义上讲，特征选择与降维技术有相似的动机。事实上它们是处理高维数据的两大主流技术。

   * 其次，去除不相关特征往往会降低学习任务的难度。

2. 特征选择过程必须确保不丢失重要特征，否则后续学习过程会因为重要信息的缺失而无法获得很好的性能。
   * 给定数据集，如果学习任务不同，则相关特征很可能不同，因此特征选择中的无关特征指的是与当前学习任务无关的特征。
   * 有一类特征称作冗余特征`redundant feature`，它们所包含的信息能从其他特征中推演出来。

     * 冗余特征在很多时候不起作用，去除它们能够减轻学习过程的负担。
     * 但如果冗余特征恰好对应了完成学习任务所需要的某个中间概念，则该冗余特征是有益的，能降低学习任务的难度。

     这里暂且不讨论冗余特征，且假设初始的特征集合包含了所有的重要信息。
3. 要想从初始的特征集合中选取一个包含了所有重要信息的特征子集，如果没有任何领域知识作为先验假设，则只能遍历所有可能的特征组合。

   这在计算上是不可行的，因为这样会遭遇组合爆炸，特征数量稍多就无法进行。

   一个可选的方案是：

   * 产生一个候选子集，评价出它的好坏。
   * 基于评价结果产生下一个候选子集，再评价其好坏。
   * 这个过程持续进行下去，直至无法找到更好的后续子集为止。

   这里有两个问题：如何根据评价结果获取下一个候选特征子集？如何评价候选特征子集的好坏？

**子集搜索**

1. 如何根据评价结果获取下一个候选特征子集？这是一个子集搜索`subset search`问题。
2. 解决该问题的算法步骤如下：
   * 给定特征集合 $$\mathbb A = \{A_1,A_2,\cdots,A_d\}$$ ，首先将每个特征看作一个候选子集（即每个子集中只有一个元素），然后对这 $$d$$ 个候选子集进行评价。

     假设 ****$$A_2$$  最优，于是将 $$A_2$$ 作为第一轮的选定子集。

   * 然后在上一轮的选定子集中加入一个特征，构成了包含两个特征的候选子集。

     假定 $$A_2,A_5$$  最优，且优于 $$A_2$$  ，于是将 $$A_2,A_5$$  作为第二轮的选定子集。

   * ....
   * 假定在第 $$k+1$$  轮时，本轮的最优的特征子集不如上一轮的最优的特征子集，则停止生成候选子集，并将上一轮选定的特征子集作为特征选择的结果。
3. 这种逐渐增加相关特征的策略称作前向`forward`搜索。

   类似地，如果从完整的特征集合开始，每次尝试去掉一个无关特征，这样逐渐减小特征的策略称作后向`backward`搜索。

4. 也可以将前向和后向搜索结合起来，每一轮逐渐增加选定相关特征（这些特征在后续迭代中确定不会被去除）、同时减少无关特征，这样的策略被称作双向`bidirectional`搜索。
5. 该策略是贪心的，因为它们仅仅考虑了使本轮选定集最优。但是除非进行穷举搜索，否则这样的问题无法避免。

#### **子集评价**

1. 如何评价候选特征子集的好坏？这是一个子集评价`subset evaluation`问题。
2. 给定数据集$$D$$ ，假设所有属性均为离散型。对属性子集$$A$$  ， 假定根据其**取值**将$$D$$分成了$$V$$个子集： $$\{D_1,D_2,...,D_v\}$$ 

   于是可以计算属性子集$$A$$ 的信息增益：$$g(D,A) = H(D) - H(D|A) = H(D) - \sum_{v=1}^V \frac{|D_v|}{|D|}H(D_v)$$

   其中$$|.|$$  为集合大小，$$H(.)$$  为熵。

   信息增益越大，则表明特征子集 $$A$$包含的有助于分类的信息越多。于是对于每个候选特征子集，可以基于训练数据集 $$D$$ 来计算其信息增益作为评价准则。

3. 更一般地，特征子集$$A$$ 实际上确定了对数据集$$D$$  的一个划分规则。
   * 每个划分区域对应着$$A$$上的一个取值，而样本标记信息$$y$$  则对应着$$D$$  的真实划分。
   * 通过估算这两种划分之间的差异，就能对$$A$$  进行评价：与$$y$$  对应的划分的差异越小，则说明  越好。
   * 信息熵仅仅是判断这个差异的一种方法，其他能判断这两个划分差异的机制都能够用于特征子集的评价。
4. 将特征子集搜索机制与子集评价机制结合就能得到特征选择方法。
   * 事实上，**决策树可以用于特征选择**，所有树结点的划分属性所组成的集合就是选择出来的特征子集。
   * 其他特征选择方法本质上都是显式或者隐式地结合了某些子集搜索机制和子集评价机制。
5. 常见的特征选择方法大致可分为三类：过滤式`filter`、包裹式`wrapper`、嵌入式`embedding` 。

#### 过滤式选择

1. 过滤式方法先对数据集进行特征选择，然后再训练学习器，特征选择过程与后续学习器无关。

   这相当于先用特征选择过程对初始特征进行过滤，再用过滤后的特征来训练模型。

2. `Relief:Relevant Features`是一种著名的过滤式特征选择方法，该方法设计了一个相关统计量来度量特征的重要性。
   * 该统计量是一个向量，其中每个分量都对应于一个初始特征。特征子集的重要性则是由该子集中每个特征所对应的相关统计量分量之和来决定的。
   * 最终只需要指定一个阈值$$\tau$$，然后选择比$$\tau$$大的相关统计量分量所对应的特征即可。

     也可以指定特征个数 $$k$$ ，然后选择相关统计量分量最大的 $$k$$个特征。
3. 给定训练集 $$\mathbb D=\{(\mathbf{\vec x}_1,\tilde y_1),(\mathbf{\vec x}_2,\tilde y_2),\cdots,(\mathbf{\vec x}_N,\tilde y_N)\},\tilde y_i\in \{0,1\}$$  。 对于每个样本 $$\mathbf{\vec x}_i$$  :  


   * `Relief` 先在 $$\mathbf{\vec x}_i$$ 同类样本中寻找其最近邻 $$\mathbf{\vec x}_{nh_i}$$ ，称作猜中近邻`near-hit` 。
   * 然后从 $$\vec x_i$$ 的异类样本中寻找其最近邻 $$\mathbf{\vec x}_{nm_i}$$，称作猜错近邻`near-miss` 。
   * 然后相关统计量对应于属性$$j$$ 的分量为： $$\delta_j=\sum_{i=1}^{N}\left(-\text{diff}(x_{i,j},x_{nh_i,j})^2+\text{diff}(x_{i,j},x_{nm_i,j})^{2}\right)$$ 

   其中 $$\text{diff}(x_{a,j},x_{b,j})$$  为两个样本在属性 $$j$$ 上的差异值，其结果取决于该属性是离散的还是连续的：

   * 如果属性$$j$$ 是离散的，则： $$\text{diff}(x_{a,j},x_{b,j})=\begin{cases} 0,&\text{if}\quad  x_{a,j}=x_{b,j}\\ 1,&else \end{cases}$$ 
   * 如果属性 $$j$$ 是连续的，则： $$\text{diff}(x_{a,j},x_{b,j})=| x_{a,j}-x_{b,j}|$$ 

     注意：此时 $$ x_{a,j},x_{b,j}$$  需要标准化到 $$[0,1]$$ 区间。

4. 从公式 $$\delta_j=\sum_{i=1}^{N}\left(-\text{diff}(x_{i,j},x_{nh_i,j})^2+\text{diff}(x_{i,j},x_{nm_i,j})^{2}\right)$$ 

   可以看出：

   * 如果 $$\mathbf{\vec x}_i$$ 与其猜中近邻 $$\mathbf{\vec x}_{nh_i}$$ 在属性 $$j$$ 上的距离小于 $$\mathbf{\vec x}_i$$ 与其猜错近邻 $$\mathbf{\vec x}_{nm_i}$$ 的距离，则说明属性 $$j$$  对于区分同类与异类样本是有益的，于是增大属性 $$j$$ 所对应的统计量分量。
   * 如果 $$\mathbf{\vec x}_i$$  与其猜中近邻 $$\mathbf{\vec x}_{nh_i}$$ 在属性 $$j$$ 上的距离大于 $$\mathbf{\vec x}_i$$ 与其猜错近邻 $$\mathbf{\vec x}_{nm_i}$$ 的距离，则说明属性$$j $$对于区分同类与异类样本是起负作用的，于是减小属性$$j$$ 所对应的统计量分量。
   * 最后对基于不同样本得到的估计结果进行平均，就得到各属性的相关统计量分量。分量值越大，则对应属性的分类能力越强。

5. `Relief` 是为二分类问题设计的，其扩展变体 `Relief-F` 能处理多分类问题。

   假定数据集$$D$$ 中的样本类别为： $$c_1,c_2,\cdots,c_K$$  。对于样本 $$\mathbf{\vec x}_i$$ ，假设 $$\tilde y_i=c_k$$ 。

   * `Relief-F` 先在类别 $$c_k$$ 的样本中寻找 $$\mathbf{\vec x}_i$$  的最近邻 $$\mathbf{\vec x}_{nh_i}$$  作为猜中近邻。
   * 然后在 $$c_k$$ 之外的每个类别中分别找到一个 $$\mathbf{\vec x}_i$$ 的最近邻 $$\mathbf{\vec x}_{nm_i^l} ,l=1,2,\cdots,K;l\ne k$$ 作为猜错近邻。
   * 于是相关统计量对应于属性 $$j $$  的分量为： $$\delta_j=\sum_{i=1}^{N}\left(-\text{diff}(x_{i,j},x_{nh_i,j})^{2}+\sum_{l\ne k}\left(p_l\times\text{diff}(x_ {i,j},x_{nm_i^l,j})^{2}\right)\right)$$ 

     其中 $$p_l$$  为第$$l$$  类的样本在数据集 $$D$$ 中所占的比例。

#### 包裹式选择

1. 与过滤式特征选择不考虑后续学习器不同，包裹式特征选择直接把最终将要使用的学习器的性能作为特征子集的评价准则。其目的就是为给定学习器选择最有利于其性能、量身定做的特征子集。
   * 优点：由于直接针对特定学习器进行优化，因此从最终学习器性能来看，效果比过滤式特征选择更好。
   * 缺点：需要多次训练学习器，因此计算开销通常比过滤式特征选择大得多。
2. `LVW:Las Vegas Wrapper`是一个典型的包裹式特征选择方法。它是`Las Vegas method`框架下使用随机策略来进行子集搜索，并以最终分类器的误差作为特征子集的评价标准。
3. `LVW`算法： 
   * 输入：
     * 数据集 $$\mathbb D=\{(\mathbf{\vec x}_1,\tilde y_1),(\mathbf{\vec x}_2,\tilde y_2),\cdots,(\mathbf{\vec x}_N,\tilde y_N)\}$$ 
     * 特征集 $$\mathbb A=\{1,2,\cdots,n\}$$ 
     * 学习器 `estimator`
     * 迭代停止条件 $$T$$ 
   * 输出： 最优特征子集 $$\mathbb A^{*}$$ 
   * 算法步骤：
     * 初始化：令候选的最优特征子集 $$\tilde {\mathbb A}^{*}=\mathbb A$$ ，然后学习器 `estimator`在特征子集 $$\tilde {\mathbb A}^{*}$$ 上使用交叉验证法进行学习，通过学习结果评估学习器 `estimator`的误差 $$err^{*}$$ 。
     * 迭代，停止条件为迭代次数到达 $$T$$ 。迭代过程为：
       * 随机产生特征子集 $$\mathbb A^{\prime}$$ 。
       * 学习器 `estimator`在特征子集 $$\mathbb A^{\prime}$$ 上使用交叉验证法进行学习，通过学习结果评估学习器 `estimator`的误差  $$err^{\prime}$$ 。
       * 如果 $$err^{\prime}$$ 比 $$err^{*}$$ 更小，或者 $$err^{\prime}=err^{*}$$  但是 $$\mathbb A^{\prime}$$ 的特征数量比 $$\tilde {\mathbb A}^{*}$$ 的特征数量更少，则将 $$\mathbb A^{\prime}$$ 作为候选的最优特征子集： $$\tilde {\mathbb A}^{*}=\mathbb A^{\prime};\quad err^{*}= err^\prime$$ 。
     * 最终 $$\mathbb A^{*}=\tilde {\mathbb A}^{*}$$ 。
4. 由于`LVW`算法中每次特征子集评价都需要训练学习器，计算开销很大，因此算法设置了停止条件控制参数 $$T$$ 。

   但是如果初始特征数量很多、 $$T$$ 设置较大、以及每一轮训练的时间较长， 则很可能算法运行很长时间都不会停止。即：如果有运行时间限制，则有可能给不出解。

#### 嵌入式选择

1. 在过滤式和包裹式特征选择方法中，特征选择过程与学习器训练过程有明显的分别。

   嵌入式特征选择是将特征选择与学习器训练过程融为一体，两者在同一个优化过程中完成的。即学习器训练过程中自动进行了特征选择。

2. 以线性回归模型为例。

   给定数据集 $$\mathbb D=\{(\mathbf{\vec x}_1,\tilde y_1),(\mathbf{\vec x}_2,\tilde y_2),\cdots,(\mathbf{\vec x}_N,\tilde y_N)\}, \tilde y_i \in \mathbb R$$ 。 以平方误差为损失函数，则优化目标为： $$\min_{\mathbf{\vec w}} \sum_{i=1}^{N}(\tilde y_i-\mathbf{\vec w}^{T}\mathbf{\vec x}_i)^{2}$$ 

   * 如果使用 $$L_2$$ 范数正则化，则优化目标为： $$\min_{\mathbf{\vec w}} \sum_{i=1}^{N}(\tilde y_i-\mathbf{\vec w}^{T}\mathbf{\vec x}_i)^{2}+\lambda||\mathbf{\vec w}||_2^{2},\quad\lambda\gt 0$$ 

     此时称作岭回归`ridge regression` \`。

   * 如果使用 $$L_1$$ 范数正则化，则优化目标为： $$\min_{\mathbf{\vec w}} \sum_{i=1}^{N}(\tilde y_i-\mathbf{\vec w}^{T}\mathbf{\vec x}_i)^{2}+\lambda||\mathbf{\vec w}||_2^{2},\quad\lambda\gt 0$$ 

     此时称作`LASSO:Least Absolute Shrinkage and Selection Operator` 回归。

3. 引入 $$L_1$$ 范数除了降低过拟合风险之外，还有一个好处：它求得的  会有较多的分量为零。即：它更容易获得稀疏解。

   于是基于 $$L_1$$  正则化的学习方法就是一种嵌入式特征选择方法，其特征选择过程与学习器训练过程融为一体，二者同时完成。

4.  正则化问题的求解可以用近端梯度下降`Proximal Gradient Descent:PGD`算法求解。

   对于优化目标： $$\min_{\mathbf{\vec x}} f(\mathbf{\vec x})+\lambda||\mathbf{\vec x}||_1$$ ，若 $$f(\mathbf{\vec x})$$ 可导且 $$\nabla f$$ 满足`L-Lipschitz`条件，即存在常数 $$L$$ 使得： $$||\nabla f(\mathbf{\vec x})-\nabla f(\mathbf{\vec x}^{\prime})||_2^{2} \le L ||\mathbf{\vec x}-\mathbf{\vec x}^{\prime} ||_2^{2},\;\forall( \mathbf{\vec x},\mathbf{\vec x}^{\prime})$$ 

   则在 $$\mathbf{\vec x}_0$$  附近将 $$ f(\mathbf{\vec x})$$ 通过二阶泰勒公式展开的近似值为： $$\hat f(\mathbf{\vec x})\simeq f(\mathbf{\vec x}_0)+\nabla f(\mathbf{\vec x}_0)\cdot (\mathbf{\vec x}-\mathbf{\vec x}_0)+\frac L2|| \mathbf{\vec x}-\mathbf{\vec x}_0||_2^{2}\\ =\frac L2||\mathbf{\vec x}- (\mathbf{\vec x}_0-\frac 1L\nabla f(\mathbf{\vec x}_0) ) ||_2^{2}+const$$ 

   其中 $$const$$ 是与 $$\mathbf{\vec x}$$ 无关的常数项。

   * 若通过梯度下降法对 $$ f(\mathbf{\vec x})$$ 进行最小化，则每一步梯度下降迭代实际上等价于最小化二次函数 。
   * 同理，若通过梯度下降法对 $$f(\mathbf{\vec x})+\lambda||\mathbf{\vec x}||_1$$ 进行最小化，则每一步梯度下降迭代实际上等价于最小化函数： $$\hat f(\mathbf{\vec x})+\lambda||\mathbf{\vec x}||_1$$ 。

     则每一步迭代为： $$\mathbf{\vec x}^{<k+1>}= \arg\min_{\mathbf{\vec x}}\frac L2||\mathbf{\vec x}- (\mathbf{\vec x}^{<k>}-\frac 1L\nabla f(\mathbf{\vec x}^{<k>}) ) ||_2^{2} +\lambda||\mathbf{\vec x} ||_1$$ 

     其中 $$\mathbf{\vec x}^{<k>}$$ 为 $$\mathbf{\vec x}$$  的第$$k$$  次迭代的值。

     该问题有解析解，因此通过`PGD`能够使得`LASSO`和其他基于$$L_1$$  范数最小化的方法能够快速求解。

5. 常见的嵌入式选择模型：
   * 在`Lasso`中， $$\lambda$$参数控制了稀疏性：
     * 如果 $$\lambda$$ 越小，则稀疏性越小，则被选择的特征越多。
     * 如果 $$\lambda$$ 越大，则稀疏性越大，则被选择的特征越少。
   * 在`SVM`和`logistic-regression`中，参数`C`控制了稀疏性
     * 如果`C`越小，则稀疏性越大，则被选择的特征越少。
     * 如果`C`越大，则稀疏性越小，则被选择的特征越多。

### 稀疏表示和字典学习

1. 对于 $$\mathbb D=\{(\mathbf{\vec x}_1,\tilde y_1),(\mathbf{\vec x}_2,\tilde y_2),\cdots,(\mathbf{\vec x}_N,\tilde y_N)\}, \tilde y_i \in \mathbb R$$ 。构建矩阵 $$\mathbf D=(\mathbf{\vec x}_1,\mathbf{\vec x}_2,\cdots,\mathbf{\vec x}_N)^{T}$$  ，其内容为： $$\mathbf D=\begin{bmatrix} x_{1,1}&x_{1,2}&\cdots&x_{1,n}\\ x_{2,1}&x_{2,2}&\cdots&x_{2,n}\\ \vdots&\vdots&\ddots&\vdots\\ x_{N,1}&x_{N,2}&\cdots&x_{N,n}\\ \end{bmatrix}$$ 

   其中每一行对应一个样本，每一列对应一个特征。

2. 特征选择所考虑的问题是：矩阵$$D$$中的许多列与当前学习任务无关。

   通过特征选择去除这些列，则学习器训练过程仅需要在较小的矩阵上进行。这样学习任务的难度可能有所降低，涉及的计算和存储开销会减少，学得模型的可解释性也会提高。

3. 考虑另一种情况：$$D$$  中有大量元素为 0 ，这称作稀疏矩阵。

   当数据集具有这样的稀疏表达形式时，对学习任务来讲会有不少好处：

   * 如果数据集具有高度的稀疏性，则该问题很可能是线性可分的。对于线性支持向量机，这种情况能取得更佳的性能。
   * 稀疏样本并不会造成存储上的巨大负担，因为稀疏矩阵已经有很多很高效的存储方法。

4. 现在问题是：如果给定数据集  $$D$$是稠密的（即普通的、非稀疏的），则能否将它转化为稀疏的形式？

   这就是字典学习 `dictionary learning`和稀疏编码`sparse coding`的目标。

#### 原理

1. 字典学习：学习一个字典，通过该字典将样本转化为合适的稀疏表示形式。它侧重于学得字典的过程。

   稀疏编码：获取样本的稀疏表达，不一定需要通过字典。它侧重于对样本进行稀疏表达的过程。

   这两者通常是在同一个优化求解过程中完成的，因此这里不做区分，统称为字典学习。

2. 给定数据集 $$\mathbb D=\{(\mathbf{\vec x}_1,\tilde y_1),(\mathbf{\vec x}_2,\tilde y_2),\cdots,(\mathbf{\vec x}_N,\tilde y_N)\}$$ ，希望对样本 $$\mathbf{\vec x}_i$$ 学习到它的一个稀疏表示 $$\vec \alpha_i\in \mathbb R^{k}$$ 。其中 $$\vec \alpha_i$$  是一个 $$k$$ 维列向量，且其中大量元素为 0 。

   一个自然的想法进行线性变换，即寻找一个矩阵 $$\mathbf P \in \mathbb R^{k\times n}$$  使得 $$\mathbf P\mathbf{\vec x}_i=\vec \alpha_i$$  。

3. 现在的问题是：既不知道变换矩阵 $$\mathbf P$$ ，也不知道 $$\mathbf{\vec x}_i$$ 的稀疏表示 $$\vec \alpha_i$$ 。

   因此求解的目标是：  


   * 根据 $$\vec \alpha_i$$ 能正确还原 $$\mathbf{\vec x}_i$$ ，或者还原的误差最小。
   *  $$\vec \alpha_i$$ 尽量稀疏，即它的分量尽量为零。

   因此给出字典学习的最优化目标： $$\min_{\mathbf B,\vec{\alpha}_i}\sum_{i=1}^{N} ||\mathbf{\vec x}_i-\mathbf B\vec{\alpha}_i||_2^{2}+\lambda\sum_{i=1}^{N}||\vec \alpha_i||_1$$ 

   其中 $$\mathbf B\in \mathbb R^{n\times k}$$ 称作字典矩阵。 $$k$$ 称作字典的词汇量，通常由用户指定来控制字典的规模，从而影响到稀疏程度。

   * 上式中第一项希望 $$\vec \alpha_i$$  能够很好地重构 $$\mathbf{\vec x}_i$$ 。
   * 第二项则希望$$ \vec \alpha_i $$尽可能稀疏。

#### 算法

1. 求解该问题采用类似`LASSO`的解法，但是使用变量交替优化的策略： 
   * 第一步：固定字典 $$B$$， 为每一个样本 $$\mathbf{\vec x}_i$$  找到相应的 $$\vec \alpha_i$$ ： $$\min_{\vec \alpha_i} ||\mathbf{\vec x}_i-\mathbf B\vec{\alpha}_i||_2^{2}+\lambda\sum_{i=1}^{N}||\vec \alpha_i||_1$$ 
   * 第二步：根据下式，以 $$\vec \alpha_i$$ 为初值来更新字典 $$B$$ ，即求解： $$\min_{\mathbf B}||\mathbf X-\mathbf B\mathbf A||_F^{2}$$ 。

     其中 $$\mathbf X=(\mathbf{\vec x}_1,\mathbf{\vec x}_2,\cdots,\mathbf{\vec x}_N)\in \mathbb R^{n\times N}$$ ， $$\mathbf A=(\vec \alpha_1,\vec \alpha_2,\cdots,\vec \alpha_N) \in  \mathbb R^{k\times N}$$  。写成矩阵的形式为： $$\mathbf X=\begin{bmatrix}   x_{1,1}&x_{2,1}&\cdots&x_{N,1}\\   x_{1,2}&x_{2,2}&\cdots&x_{N,2}\\   \vdots&\vdots&\ddots&\vdots\\   x_{1,n}&x_{2,n}&\cdots&x_{N,n}   \end{bmatrix}\quad   \mathbf A=\begin{bmatrix}   \alpha_{1,1}&\alpha_{2,1}&\cdots&\alpha_{N,1}\\   \alpha_{1,2}&\alpha_{2,2}&\cdots&\alpha_{N,2}\\   \vdots&\vdots&\ddots&\vdots\\   \alpha_{1,n}&\alpha_{2,n}&\cdots&\alpha_{N,n}   \end{bmatrix}\\$$ 

     这里 $$||\cdot||_F$$ 为矩阵的 `Frobenius`范数（所有元素的平方和的平方根）。对于矩阵 $$M$$  ， 有 $$||\mathbf M ||_F=\sqrt{\sum_{i}\sum_{j}|m_{ij}|^{2}}$$ 

   * 反复迭代上述两步，最终即可求得字典  和样本  的稀疏表示 。
2. 这里有个最优化问题： $$\min_{\mathbf B}||\mathbf X-\mathbf B\mathbf A||_F^{2} $$ 

   该问题有多种求解方法，常用的有基于逐列更新策略的`KSVD`算法。

   令 $$\mathbf {\vec b}_i$$ 为字典矩阵 $$B$$  的第 $$i$$ 列，  $$\mathbf{\vec a}^{j}$$ 表示稀疏矩阵 $$A$$ 的第 $$j$$ 行。 固定 $$B$$ 其他列，仅考虑第 $$i$$  列，则有： $$\min_{\mathbf {\vec b}_i}||\mathbf X-\sum_{j=1}^{k}\mathbf {\vec b}_i\mathbf{\vec a}^{j}||_F^{2} =\min_{\mathbf {\vec b}_i}||(\mathbf X-\sum_{j=1,j\ne i}^{k}\mathbf {\vec b}_i\mathbf{\vec a}^{j})-\mathbf {\vec b}_i\mathbf{\vec a}^{i}||_F^{2}$$ 

   令 $$\mathbf E_i=\mathbf X-\sum_{j=1,j\ne i}^{k}\mathbf {\vec b}_i\mathbf{\vec a}^{j}$$  ，它表示去掉 $$\mathbf{\vec x}_i$$ 的稀疏表示之后，样本集的稀疏表示与原样本集的误差矩阵。

   考虑到更新字典的第 $$i$$ 列 $$\mathbf {\vec b}_i$$ 时，其他各列都是固定的，则 $$\mathbf E_i$$ 是固定的。则最优化问题转换为： $$\min_{\mathbf {\vec b}_i}||\mathbf E_i-\mathbf {\vec b}_i\mathbf{\vec a}^{i} ||_F^{2}$$ 求解该最优化问题只需要对 $$\mathbf E_i$$ 进行奇异值分解以取得最大奇异值所对应的正交向量。

3. 直接对 $$\mathbf E_i$$  进行奇异值分解会同时修改 $$\mathbf {\vec b}_i$$ 和 $$\mathbf{\vec a}^{i}$$ ， 从而可能破坏$$A$$ 的稀疏性。因为第二步 “以 $$\vec \alpha_i$$ 为初值来更新字典 $$B$$  ” 中， 在更新 $$B$$ 前后$$ \vec \alpha_i $$的非零元所处的位置和非零元素的值很可能不一致。

   为避免发生这样的情况 `KSVD` 对 $$\mathbf E_i$$ 和 $$\mathbf{\vec a}^{i}$$ 进行了专门处理：  


   *  $$\mathbf{\vec a}^{i}$$ 仅保留非零元素。
   *  $$\mathbf E_i$$ 仅保留 $$\mathbf {\vec b}_i$$  和 $$\mathbf{\vec a}^{i}$$ 的非零元素的乘积项，然后再进行奇异值分解，这样就保持了第一步得到的稀疏性。


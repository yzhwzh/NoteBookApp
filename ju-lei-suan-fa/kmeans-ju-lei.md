# K-means 聚类

## **K-means**

1. 给定样本集 $$\mathbb D=\{\mathbf{\vec x}_1,\mathbf{\vec x}_2,\cdots,\mathbf{\vec x}_N\}$$ ， 假设一个划分为 $$\mathcal C=\{\mathbb C_1,\mathbb C_2,\cdots,\mathbb C_K\}$$ 。

   定义该划分的平方误差为： $$err=\sum_{k=1}^{K}\sum_{\mathbf{\vec x}_i \in \mathbb C_k}||\mathbf{\vec x}_i-\vec \mu_k||_2^{2}$$ 

   其中 $$\vec \mu_k=\frac {1}{|\mathbb C_k|}\sum_{\mathbf{\vec x}_i \in \mathbb C_k}\mathbf{\vec x}_i$$  是簇  $$\mathbb C_k$$ 的均值向量。

   *  $$err$$ 刻画了簇类样本围绕簇均值向量的紧密程度，其值越小，则簇内样本相似度越高。
   * `k-means` 算法的优化目标为：最小化  。即 $$\min_{\mathcal C} \sum_{k=1}^{K}\sum_{\mathbf{\vec x}_i \in C_k}||\mathbf{\vec x}_i-\vec \mu_k||_2^{2}$$ ： 。

2. `k-means` 的优化目标需要考察 $$\mathbb D$$ 的所有可能的划分，这是一个`NP`难的问题。实际上`k-means` 采用贪心策略，通过迭代优化来近似求解。
   * 首先假设一组均值向量。
   * 然后根据假设的均值向量给出了 $$\mathbb D$$ 的一个划分。
   * 再根据这个划分来计算真实的均值向量：
     * 如果真实的均值向量等于假设的均值向量，则说明假设正确。根据假设均值向量给出的 $$\mathbb D$$ 的一个划分确实是原问题的解。
     * 如果真实的均值向量不等于假设的均值向量，则可以将真实的均值向量作为新的假设均值向量，继续迭代求解。
3. 这里的一个关键就是：给定一组假设的均值向量，如何计算出 $$\mathbb D$$ 的一个簇划分？

   `K`**均值算法的策略是：样本离哪个簇的均值向量最近，则该样本就划归到那个簇。**

4. `k-means` 算法：
   * 输入：
     * 样本集 $$\mathbb D=\{\mathbf{\vec x}_1,\mathbf{\vec x}_2,\cdots,\mathbf{\vec x}_N\}$$ 。
     * 聚类簇数 $$K$$ 。
   * 输出：簇划分 $$\mathcal C=\{\mathbb C_1,\mathbb C_2,\cdots,\mathbb C_K\}$$ 。
   * 算法步骤：
     * 从  $$\mathbb D$$ 中随机选择 $$K$$ 个样本作为初始均值向量 $$\{\vec \mu_1,\vec \mu_2,\cdots,\vec \mu_K\}$$ 。
     * 重复迭代直到算法收敛，迭代过程：
       * 初始化阶段：取 $$\mathbb C_k=\phi,k=1,2,\cdots,K$$ 
       * 划分阶段：令  $$i=1,2,\cdots,N$$ ：
         * 计算  $$\mathbf{\vec x}_i$$ 的簇标记： $$\lambda_i=\arg\min_{k}||\mathbf{\vec x}_i-\vec \mu_k||_2 ,k \in \{1,2,\cdots,K\}$$ 。

           即：将 $$\mathbf{\vec x}_i$$ 离哪个簇的均值向量最近，则该样本就标记为那个簇。

         * 然后将样本 $$\mathbf{\vec x}_i$$ 划入相应的簇： $$\mathbb C_{\lambda_i}= \mathbb C_{\lambda_i} \bigcup\{\mathbf{\vec x}_i\}$$ 。
       * 重计算阶段：计算 $$\hat{\vec \mu}_k $$ ： $$\hat{\vec \mu}_k =\frac {1}{|\mathbb C_k|}\sum_{\mathbf{\vec x}_i \in \mathbb C_k}\mathbf{\vec x}_i$$ 。
       * 终止条件判断：
         * 如果对所有的 $$k \in \{1,2,\cdots,K\}$$ ，都有 $$\hat{\vec \mu}_k=\vec \mu_k$$ ，则算法收敛，终止迭代。
         * 否则重赋值  。
5. `k-means` 优点：
   * 计算复杂度低，为  $$O(N\times K\times q)$$ ，其中  $$q$$ 为迭代次数。

     通常 $$K$$ 和 $$q$$ 要远远小于 $$N$$ ，此时复杂度相当于 $$O(N)$$ 。

   * 思想简单，容易实现。
6. `k-means` 缺点：
   * 需要首先确定聚类的数量 $$K$$ 。
   * 分类结果严重依赖于分类中心的初始化。

     通常进行多次`k-means`，然后选择最优的那次作为最终聚类结果。

   * 结果不一定是全局最优的，只能保证局部最优。
   * 对噪声敏感。因为簇的中心是取平均，因此聚类簇很远地方的噪音会导致簇的中心点偏移。
   * 无法解决不规则形状的聚类。
   * 无法处理离散特征，如：`国籍、性别` 等。
7. `k-means` 性质：
   * `k-means` 实际上假设数据是呈现球形分布，实际任务中很少有这种情况。

     与之相比，`GMM` 使用更加一般的数据表示，即高斯分布。

   * `k-means` 假设各个簇的先验概率相同，但是各个簇的数据量可能不均匀。
   * `k-means` 使用欧式距离来衡量样本与各个簇的相似度。这种距离实际上假设数据的各个维度对于相似度的作用是相同的。
   * `k-means` 中，各个样本点只属于与其相似度最高的那个簇，这实际上是`硬` 分簇。 
   * `k-means` 算法的迭代过程实际上等价于`EM` 算法。具体参考`EM` 算法章节。

## **k-means++**

1. `k-means++` 属于 `k-means` 的变种，它主要解决`k-means` 严重依赖于分类中心初始化的问题。
2. `k-means++` 选择初始均值向量时，尽量安排这些初始均值向量之间的距离尽可能的远。
3. `k-means++` 算法：

   * 输入：
     * 样本集 $$\mathbb D=\{\mathbf{\vec x}_1,\mathbf{\vec x}_2,\cdots,\mathbf{\vec x}_N\}$$ 。
     * 聚类簇数 $$K$$ 。
   * 输出：簇划分 $$\mathcal C=\{\mathbb C_1,\mathbb C_2,\cdots,\mathbb C_K\}$$ 。
   * 算法步骤：

     * 从 $$\mathbb D$$ 中随机选择1个样本作为初始均值向量组  $$\{\vec \mu_1,\}$$ 。
     * 迭代，直到初始均值向量组有 $$K$$ 个向量。

       假设初始均值向量组为 $$\{\vec\mu_1,\cdots,\vec\mu_m\}$$ 。迭代过程如下：

       * 对每个样本 $$\mathbf{\vec x}_i$$ ，分别计算其距 $$\vec\mu_1,\cdots,\vec\mu_m$$  的距离。这些距离的最小值记做 $$d_i=\min_{\vec\mu_j} ||\mathbf{\vec x}_i-\vec\mu_j||$$  。
       * 对样本 $$\mathbf{\vec x}_i$$ ，其设置为初始均值向量的概率正比于 $$d_i$$ 。即：离所有的初始均值向量越远，则越可能被选中为下一个初始均值向量。
       * 以概率分布  $$P=\{d_1,d_2,\cdots,d_N\}$$ （未归一化的）随机挑选一个样本作为下一个初始均值向量 $$\vec\mu_{m+1}$$ 。

     * 一旦挑选出初始均值向量组 ，剩下的迭代步骤与`k-means` 相同。

   > 详细解释，例子

![](../.gitbook/assets/image%20%2825%29.png)

![](../.gitbook/assets/image%20%2823%29.png)

![](../.gitbook/assets/image%20%2824%29.png)

## **K-modes**

1. `k-modes` 属于 `k-means` 的变种，它主要解决`k-means` **无法处理离散特征的问题**。
2. `k-modes` 与`k-means` 有两个不同点（假设所有特征都是离散特征）：
   * 距离函数不同。在`k-modes` 算法中，距离函数为： $$\text{distance}( \mathbf {\vec x}_i, \mathbf {\vec x}_j )=\sum_{d=1}^n I(x_{i,d} \ne x_{j,d})$$ 

     其中 $$I(\cdot)$$ 为示性函数。

     上式的意义为：**样本之间的距离等于它们之间不同属性值的个数**。

   * 簇中心的更新规则不同。在`k-modes` 算法中，**簇中心每个属性的取值为：簇内该属性出现频率最大的那个值**。 $$\hat\mu_{k,d} = \arg\max_{v} \sum_{\mathbf{\vec x}_i\in \mathbb C_k} I(x_{i,d}=v)$$ 

     其中 $$v$$ 的取值空间为所有样本在第 $$d$$ 个属性上的取值。

   * 将样本划分到距离最小的簇，在全部的样本都被划分完毕之后，重复以上步骤，直到总距离（各个簇中样本与各自簇中心距离之和）不再降低，返回最后的聚类结果

## **K-medoids**

1. `k-medoids` 属于 `k-means` 的变种，它主要解决`k-means` 对噪声敏感的问题。
2. `k-medoids` 算法：
   * 输入：
     * 样本集 $$\mathbb D=\{\mathbf{\vec x}_1,\mathbf{\vec x}_2,\cdots,\mathbf{\vec x}_N\}$$ 。
     * 聚类簇数 $$K$$ 。
   * 输出：簇划分 $$\mathcal C=\{\mathbb C_1,\mathbb C_2,\cdots,\mathbb C_K\}$$ 。
   * 算法步骤：
     * 从 $$\mathbb D$$ 中随机选择 $$K$$ 个样本作为初始均值向量 $$\{\vec \mu_1,\vec \mu_2,\cdots,\vec \mu_K\}$$ 。
     * 重复迭代直到算法收敛，迭代过程：
       * 初始化阶段：取 $$\mathbb C_k=\phi,k=1,2,\cdots,K$$ 。

         遍历每个样本 $$\mathbf{\vec x}_i,i=1,2,\cdots,N$$ ，计算它的簇标记： $$\lambda_i=\arg\min_{k}||\mathbf{\vec x}_i-\vec \mu_k||_2 ,k \in \{1,2,\cdots,K\}$$ 。即：将 $$\mathbf{\vec x}_i$$ 离哪个簇的均值向量最近，则该样本就标记为那个簇。

         然后将样本 $$\mathbf{\vec x}_i$$ 划入相应的簇： $$\mathbb C_{\lambda_i}= \mathbb C_{\lambda_i} \bigcup\{\mathbf{\vec x}_i\}$$ 。

       * 重计算阶段：

         遍历每个簇 $$\mathbb C_k,k=1,2,\cdots,K$$ ：

         * 计算簇心 $$\vec\mu_k$$ 距离簇内其它点的距离 $$d_\mu^{(k)}=\sum_{\mathbf{\vec x}_j^{(k)}\in \mathbb C_k}||\vec\mu_k-\mathbf{\vec x}_j^{(k)}||$$ 。
         * 计算簇 $$\mathbb C_k$$ 内每个点 $$\mathbf{\vec x}_i^{(k)}$$ 距离簇内其它点的距离 $$d_i^{(k)}=\sum_{\mathbf{\vec x}_j^{(k)}\in \mathbb C_k}||\mathbf{\vec x}_i^{(k)}-\mathbf{\vec x}_j^{(k)}||$$ 。

           如果 $$d_i^{(k)}\lt d_\mu^{(k)}$$ ，则重新设置簇中心： $$\vec\mu_k = \mathbf{\vec x}_i^{(k)},\quad d_\mu^{(k)}= d_i^{(k)}$$ 。

       * 终止条件判断：遍历一轮簇  之后，簇心保持不变。
3. `k-medoids` 算法在计算新的簇心时，不再通过簇内样本的均值来实现，而是挑选簇内距离其它所有点都最近的样本来实现。这就减少了孤立噪声带来的影响。
4. `k-medoids` 算法复杂度较高，为 $$O(N^2)$$ 。其中计算代价最高的是计算每个簇内每对样本之间的距离。

   通常会在算法开始时计算一次，然后将结果缓存起来，以便后续重复使用。

## **mini-batch K-means**

1. `mini-batch k-means` 属于 `k-means` 的变种，它主要用于减少`k-means` 的计算时间。
2. `mini-batch k-means` 算法每次训练时随机抽取小批量的数据，然后用这个小批量数据训练。这种做法减少了`k-means` 的收敛时间，其效果略差于标准算法。
3. `mini-batch k-means` 算法：
   * 输入：
     * 样本集 $$\mathbb D=\{\mathbf{\vec x}_1,\mathbf{\vec x}_2,\cdots,\mathbf{\vec x}_N\}$$ 。
     * 聚类簇数 $$K$$ 。
   * 输出：簇划分 $$\mathcal C=\{\mathbb C_1,\mathbb C_2,\cdots,\mathbb C_K\}$$ 。
   * 算法步骤：
     * 从 $$\mathbb D$$ 中随机选择 $$K$$ 个样本作为初始均值向量 $$\{\vec \mu_1,\vec \mu_2,\cdots,\vec \mu_K\}$$ 。
     * 重复迭代直到算法收敛，迭代过程：
       * 初始化阶段：取 $$\mathbb C_k=\phi,k=1,2,\cdots,K$$ 
       * 划分阶段：随机挑选一个`Batch` 的样本集合 $$\mathbb B={\mathbf{\vec x}_{b_1},\cdots,\mathbf{\vec x}_{b_M}}$$ ，其中  $$M$$ 为批大小。
         * 计算 $$\mathbf{\vec x}_i,i=b_1,\cdots,b_M$$ 的簇标记： $$\lambda_i=\arg\min_{k}||\mathbf{\vec x}_i-\vec \mu_k||_2 ,k \in \{1,2,\cdots,K\}$$ 。

           即：将 $$\mathbf{\vec x}$$ 离哪个簇的均值向量最近，则该样本就标记为那个簇。

         * 然后将样本  $$\mathbf{\vec x}$$ 划入相应的簇： $$\mathbb C_{\lambda_i}= \mathbb C_{\lambda_i} \bigcup\{\mathbf{\vec x}_i\}$$ 。
       * 重计算阶段：计算 $$\hat{\vec \mu}_k $$ ： $$\hat{\vec \mu}_k =\frac {1}{|\mathbb C_k|}\sum_{\mathbf{\vec x}_i \in \mathbb C_k}\mathbf{\vec x}_i$$ 。
       * 终止条件判断：
         * 如果对所有的 $$k \in \{1,2,\cdots,K\}$$ ，都有 $$\hat{\vec \mu}_k=\vec \mu_k$$ ，则算法收敛，终止迭代。
         * 否则重赋值 $$\vec \mu_k=\hat{\vec \mu}_k$$ 。

> 说人话：算法步骤如下: ●首先抽取部分数据集,使用K-Means算法构建出K个聚簇点的模型 ●继续抽取训\|练数据集中的部分数据集样本数据，并将其添加到模型中，分配给距离最近的聚簇中心点 ●更新聚簇的中心点值\(**每次更新都只用抽取出来的部分数据集**\) ●循环迭代第二步和第三步操作，直到中心点稳定或者达到迭代次数,停止计算操作




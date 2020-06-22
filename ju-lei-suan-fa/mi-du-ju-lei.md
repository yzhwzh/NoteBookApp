# 密度聚类

1. 密度聚类`density-based clustering`假设聚类结构能够通过样本分布的紧密程度确定。
2. 密度聚类算法从样本的密度的角度来考察样本之间的可连接性，并基于可连接样本的不断扩张聚类簇，从而获得最终的聚类结果。

## 1 DBSCAN 算法

1. `DBSCAN`是一种著名的密度聚类算法，它基于一组邻域参数 $$(\epsilon, MinPts)$$ 来刻画样本分布的紧密程度。
2. 给定数据集 $$\mathbb D=\{\mathbf{\vec x}_1,\mathbf{\vec x}_2,\cdots,\mathbf{\vec x}_N\}$$ ， 定义：
   *  $$\epsilon$$ -邻域： $$N_\epsilon(\mathbf{\vec x}_i)=\{\mathbf{\vec x}_j \in \mathbb D \mid distance(\mathbf{\vec x}_i,\mathbf{\vec x}_j) \le \epsilon \}$$ 。

     即： $$N_\epsilon(\mathbf{\vec x}_i)$$ 包含了样本集 $$\mathbb D$$ 中与 $$\mathbf{\vec x}_i$$ 距离不大于 $$\epsilon$$ 的所有的样本。

   * 核心对象`core object`：若  $$|N_\epsilon(\mathbf{\vec x}_i)|\ge MinPts$$ ，则称 $$\mathbf{\vec x}_i$$ 是一个核心对象。

     即：若 $$\mathbf{\vec x}_i$$ 的  $$\epsilon$$ -邻域中至少包含 $$MinPts$$  个样本，则 $$\mathbf{\vec x}_i$$ 是一个核心对象。

   * 密度直达`directyl density-reachable`：若 $$\mathbf{\vec x}_i$$ 是一个核心对象，且 $$\mathbf{\vec x}_j \in N_\epsilon(\mathbf{\vec x}_i) $$ ， 则称 $$\mathbf{\vec x}_j$$ 由 $$\mathbf{\vec x}_i$$  密度直达，记作 $$\mathbf{\vec x}_i \mapsto \mathbf{\vec x}_j$$ 。
   * 密度可达`density-reachable`：对于 $$\mathbf{\vec x}_i$$  和 $$\mathbf{\vec x}_j $$ ， 若存在样本序列 $$(\mathbf {\vec p}_0,\mathbf {\vec p}_1,\mathbf {\vec p}_2,\cdots,\mathbf {\vec p}_m,\mathbf {\vec p}_{m+1}) $$ ， 其中 $$\mathbf {\vec p}_0=\mathbf{\vec x}_i,\mathbf {\vec p}_{m+1} =\mathbf{\vec x}_j,\mathbf {\vec p}_s \in \mathbb D$$ ，如果 $$\mathbf {\vec p}_{s+1}$$ 由 $$\mathbf {\vec p}_s$$  密度直达，则称 $$\mathbf{\vec x}_j $$ 由 $$\mathbf{\vec x}_i$$ 密度可达，记作  $$\mathbf{\vec x}_i \leadsto \mathbf{\vec x}_j$$ 。
   * 密度相连`density-connected`：对于 $$\mathbf{\vec x}_i$$ 和 $$\mathbf{\vec x}_j $$ ，若存在 $$\mathbf{\vec x}_k$$  ，使得 $$\mathbf{\vec x}_i$$ 与 $$\mathbf{\vec x}_j $$ 均由 $$\mathbf{\vec x}_k$$ 密度可达，则称 $$\mathbf{\vec x}_i$$ 与 $$\mathbf{\vec x}_j $$ 密度相连 ，记作 $$\mathbf{\vec x}_i \sim \mathbf{\vec x}_j$$ 。
3. `DBSCAN`算法的簇定义：给定邻域参数 $$(\epsilon, MinPts)$$ ， 一个簇 $$\mathbb C \subseteq \mathbb D$$ 是满足下列性质的非空样本子集：

   * 连接性 `connectivity`： 若 $$\mathbf{\vec x}_i \in \mathbb C,\mathbf{\vec x}_j \in \mathbb C$$ ，则 $$\mathbf{\vec x}_i \sim \mathbf{\vec x}_j$$ 。
   * 最大性`maximality`：若 $$\mathbf{\vec x}_i \in \mathbb C$$ ，且 $$\mathbf{\vec x}_i \leadsto \mathbf{\vec x}_j$$ ， 则  $$\mathbf{\vec x}_j \in \mathbb C$$ 。

   即一个簇是由密度可达关系导出的最大的密度相连样本集合。

4. `DBSCAN`算法的思想：若 $$\mathbf{\vec x} $$ 为核心对象，则 $$\mathbf{\vec x} $$ 密度可达的所有样本组成的集合记作 $$\mathbb  X=\{\mathbf{\vec x}^{\prime} \in \mathbb D \mid \mathbf{\vec x} \leadsto \mathbf{\vec x}^{\prime} \}$$ 。可以证明 ： $$\mathbb X$$ 就是满足连接性与最大性的簇。

   于是 `DBSCAN`算法首先任选数据集中的一个核心对象作为种子`seed`，再由此出发确定相应的聚类簇。

5. `DBSCAN`算法：
   * 输入：
     * 数据集 $$\mathbb D=\{\mathbf{\vec x}_1,\mathbf{\vec x}_2,\cdots,\mathbf{\vec x}_N\}$$ 
     * 邻域参数 $$(\epsilon, MinPts)$$ 
   * 输出： 簇划分 $$\mathcal C=\{\mathbb C_1,\mathbb C_2,\cdots,\mathbb C_K\}$$ 
   * 算法步骤：
     * 初始化核心对象集合为空集： $$\Omega=\phi$$ 
     * 寻找核心对象：
       * 遍历所有的样本点 ， $$\mathbf{\vec x}_i,i=1,2,\cdots,N$$ 计算 $$N_\epsilon(\mathbf{\vec x}_i)$$ 
       * 如果 $$|N_\epsilon(\mathbf{\vec x}_i)|\ge MinPts$$ ，则 $$\Omega=\Omega\bigcup \{\mathbf{\vec x}_i\}$$ 
     * 迭代：以任一未访问过的核心对象为出发点，找出有其密度可达的样本生成的聚类簇，直到所有核心对象都被访问为止。
6. 注意：
   * 若在核心对象 $$\mathbf{\vec o}_1$$ 的寻找密度可达的样本的过程中，发现核心对象 $$\mathbf{\vec o}_2$$ 是由 $$\mathbf{\vec o}_1$$ 密度可达的，且  尚未被访问，则将 $$\mathbf{\vec o}_2$$ 加入 $$\mathbf{\vec o}_1$$ 所属的簇，并且标记 $$\mathbf{\vec o}_2$$ 为已访问。
   * 对于 $$\mathbb D$$ 中的样本点，它只可能属于某一个聚类簇。因此在核心对象  $$\mathbf{\vec o}_i$$ 的寻找密度可达的样本的过程中，它只能在标记为未访问的样本中寻找 （标记为已访问的样本已经属于某个聚类簇了）。
7. `DBSCAN` 算法的优点：
   * 簇的数量由算法自动确定，无需人工指定。
   * 基于密度定义，能够对抗噪音。
   * 可以处理任意形状和大小的簇。
8. `DBSCAN` 算法的缺点：
   * 若样本集的密度不均匀，聚类间距差相差很大时，聚类质量较差。因为此时参数 $$\epsilon$$ 和 $$MinPts$$ 的选择比较困难。
   * 无法应用于密度不断变化的数据集中。

## 2 Mean-Shift 算法

1. `Mean-Shift` 是基于核密度估计的爬山算法，可以用于聚类、图像分割、跟踪等领域。
2. 给定 $$n$$ 维空间的  $$N$$ 个样本组成的数据集 $$\mathbb D=\{\mathbf{\vec x}_1,\cdots,\mathbf{\vec x}_N\}$$ ，给定一个中心为 $$\mathbf{\vec x}$$ 、半径为 $$h$$ 的球形区域  $$S$$ （称作`兴趣域`），定义其`mean shift` 向量为： 。
3. `Mean-Shift` 算法的基本思路是：

   * 首先任选一个点作为聚类的中心来构造`兴趣域`。
   * 然后计算当前的`mean shift` 向量，`兴趣域`的中心移动为： $$\mathbf{\vec x}\leftarrow \mathbf{\vec x} +\mathbf{\vec M}(\mathbf{\vec x})$$ 。

     移动过程中，`兴趣域`范围内的所有样本都标记为同一个簇。

   * 如果`mean shift` 向量为 0 ，则停止移动，说明`兴趣域` 已到达数据点最密集的区域。

   因此`Mean-Shift` 会向着密度最大的区域移动。

   下图中：蓝色为当前的`兴趣域`，红色为当前的中心点 ；紫色向量为`mean shift` 向量  ，灰色为移动之后的`兴趣域` 。

   ![mean\_shift](http://www.huaxiaozhuan.com/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0/imgs/clustering/mean_shift.png)

4. 在计算`mean shift` 向量的过程中假设每个样本的作用都是相等的。实际上随着样本与中心点的距离不同，样本对于`mean shift` 向量的贡献不同。

   定义高斯核函数为： $$g(x)=\frac {1}{\sqrt{2\pi}}\exp(-\frac{x}{2})$$ ，则重新`mean shift` 向量定义为： $$\mathbf{\vec M}(\mathbf{\vec x}) = \mathbf{\vec m}(\mathbf{\vec x}) - \mathbf{\vec x},\quad \mathbf{\vec m}(\mathbf{\vec x})=\frac{\sum_{\mathbf{\vec x}_i\in \mathbb S}\mathbf{\vec x}_ig(||\frac{\mathbf{\vec x}_i-\mathbf{\vec x}}{h}||^2)} {\sum_{\mathbf{\vec x}_i\in \mathbb S}g(||\frac{\mathbf{\vec x}_i-\mathbf{\vec x}}{h}||^2)}$$ 

   其中 $$h$$ 称做带宽。 $$||\frac{\mathbf{\vec x}_i-\mathbf{\vec x}}{h}||$$ 刻画了样本  $$\mathbf{\vec x}_i$$ 距离中心点 $$\mathbf{\vec x}$$ 相对于半径 $$h$$ 的相对距离。

5. `Mean_Shift` 算法：
   * 输入：
     * 数据集 $$\mathbb D=\{\mathbf{\vec x}_1,\mathbf{\vec x}_2,\cdots,\mathbf{\vec x}_N\}$$ 
     * 带宽参数 $$h$$ 
     * 迭代阈值 $$\epsilon_1$$ ，簇阈值 $$\epsilon_2$$ 
   * 输出： 簇划分 $$\mathcal C=\{\mathbb C_1,\mathbb C_2,\cdots\}$$ 
   * 算法步骤：

     迭代，直到所有的样本都被访问过。迭代过程为（设已有的簇为 $$1,2,\cdots,L-1$$ ）：

     * 在未访问过的样本中随机选择一个点作为中心点  $$\mathbf{\vec x}$$ ，找出距它半径为 $$h$$ 的`兴趣域`，记做 $$S$$ 。

       将 $$S$$ 中的样本的簇标记设置为  $$L$$ （ 一个新的簇）。

     * 计算当前的`mean shift` 向量，兴趣域中心的移动为： $$\mathbf{\vec x}\leftarrow \mathbf{\vec x}+\mathbf{\vec M}(\mathbf{\vec x})=\mathbf{\vec m}(\mathbf{\vec x})=\frac{\sum_{\mathbf{\vec x}_i\in \mathbb S}\mathbf{\vec x}_ig(||\frac{\mathbf{\vec x}_i-\mathbf{\vec x}}{h}||^2)} {\sum_{\mathbf{\vec x}_i\in \mathbb S}g(||\frac{\mathbf{\vec x}_i-\mathbf{\vec x}}{h}||^2)}$$ 

       在移动过程中，兴趣域内的所有点标记为`访问过`，并且将它们的簇标记设置为  。

     * 如果 $$||\mathbf{\vec M}(\mathbf{\vec x})||\le \epsilon_1$$ ，则本次结束本次迭代。
     * 设已有簇中，簇 $$l$$ 的中心点 $$\mathbf{\vec x}^{(l)}$$  与 $$\mathbf{\vec x}$$ 距离最近，如果 $$||\mathbf{\vec x}^{(l)}-\mathbf{\vec x}||\le \epsilon_2$$ ，则将当前簇和簇 $$l$$ 合并。

       合并时，当前簇中的样本的簇标记重新修改为 $$l$$ 。

     当所有的样本都被访问过时，重新分配样本的簇标记（因为可能有的样本被多个簇标记过）：若样本被多个簇标记过，则选择最大的那个簇作为该样本的簇标记。即：尽可能保留大的簇。
6. 可以证明：`Mean_Shift` 算法每次移动都是向着概率密度函数增加的方向移动。

   在 $$n$$ 维欧式空间中，对空间中的点 $$\mathbf{\vec x}$$ 的概率密度估计为： $$\hat f(\mathbf{\vec x}) = \frac 1N\sum_{i=1}^N K_H(\mathbf{\vec x}-\mathbf{\vec x}_i),\quad K_H(\mathbf{\vec x})=|\mathbf H|^{-\frac 12}K(\mathbf H^{-\frac 12}\mathbf{\vec x})$$ 

   其中：

   *  $$K(\mathbf{\vec x })$$ 表示空间中的核函数， $$\mathbf H$$ 为带宽矩阵。
   * 通常 $$K(\cdot)$$ 采用放射状对称核函数 $$K(\mathbf{\vec x}) = c_k\times k(||\mathbf{\vec x}||^2)$$ ， $$k(\cdot)$$ 为 $$K(\cdot)$$ 的轮廓函数，  $$c_k$$ （一个正数）为标准化常数从而保证 $$K(\mathbf{\vec x})$$  的积分为 1 。
   * 通常带宽矩阵采用 $$\mathbf H=h^2\mathbf I$$ ， $$h$$ 为带宽参数。

   因此有： $$\hat f(\mathbf{\vec x})=\frac {c_k}{Nh^n}\sum_{k=1}^Nk(||\frac{\mathbf{\vec x}-\mathbf{\vec x}_i}{h}||^2)$$ 。则有梯度： $$\nabla_{\mathbf{\vec x}}\hat f(\mathbf{\vec x}) =\frac {2c_k}{Nh^{n+2}}\sum_{k=1}^N(\mathbf{\vec x}-\mathbf{\vec x}_i)k^\prime(||\frac{\mathbf{\vec x}-\mathbf{\vec x}_i}{h}||^2)$$ 

   记 $$k(\cdot)$$ 的导数为 $$g(\cdot)=k^\prime(\cdot)$$ 。取 $$g(\cdot)$$ 为新的轮廓函数， $$c_g$$ （一个正数）为新的标准化常数， $$G(\mathbf{\vec x})=c_g\times g(||\mathbf{\vec x}||^2)$$  。

   则有： $$\nabla_{\mathbf{\vec x}}\hat f(\mathbf{\vec x}) =\frac {2c_k}{Nh^{n+2}}\sum_{i=1}^N(\mathbf{\vec x}-\mathbf{\vec x}_i)g(||\frac{\mathbf{\vec x}-\mathbf{\vec x}_i}{h}||^2)\\ = \frac {2c_k}{h^2c_g}\left[\frac{c_g}{Nh^n}\sum_{i=1}^Ng\left(||\frac{\mathbf{\vec x}-\mathbf{\vec x}_i}{h}||^2\right)\right]\left[\frac{\sum_{i=1}^N\mathbf{\vec x}_ig\left(||\frac{\mathbf{\vec x}-\mathbf{\vec x}_i}{h}||^2\right)}{\sum_{i=1}^Ng\left(||\frac{\mathbf{\vec x}-\mathbf{\vec x}_i}{h}||^2\right)}-\mathbf{\vec x}\right]$$ 

   定义  $$\hat f_g(\mathbf{\vec x})= \frac{c_g}{Nh^n}\sum_{i=1}^Ng\left(||\frac{\mathbf{\vec x}-\mathbf{\vec x}_i}{h}||^2\right)$$ ，则它表示基于核函数 $$G(\cdot)$$ 的概率密度估计，始终为非负数。

   根据前面定义： $$\mathbf{\vec M}(\mathbf{\vec x})=\frac{\sum_{i=1}^N\mathbf{\vec x}_ig\left(||\frac{\mathbf{\vec x}-\mathbf{\vec x}_i}{h}||^2\right)}{\sum_{i=1}^Ng\left(||\frac{\mathbf{\vec x}-\mathbf{\vec x}_i}{h}||^2\right)}-\mathbf{\vec x}$$ ，则有： $$\nabla_{\mathbf{\vec x}}\hat f(\mathbf{\vec x}) = \frac {2c_k}{h^2c_g}\times\hat f_g(\mathbf{\vec x})\times \mathbf{\vec M}(\mathbf{\vec x})$$ 。

   因此 $$\mathbf{\vec M}(\mathbf{\vec x})$$ 正比于 $$\nabla_{\mathbf{\vec x}}\hat f(\mathbf{\vec x}) $$ ，因此`mean shift` 向量的方向始终指向概率密度增加最大的方向。

   > 上式计算 $$\sum_{i=1}^N$$ 时需要考虑所有的样本，计算复杂度太大。作为一个替代，可以考虑使用 $$\mathbf{\vec x}$$ 距离 $$h$$  内的样本，即`兴趣域` 内的样本。即可得到： $$\mathbf{\vec M}(\mathbf{\vec x})=\frac{\sum_{\mathbf{\vec x}_i\in \mathbb S}\mathbf{\vec x}_ig(||\frac{\mathbf{\vec x}_i-\mathbf{\vec x}}{h}||^2)} {\sum_{\mathbf{\vec x}_i\in \mathbb S}g(||\frac{\mathbf{\vec x}_i-\mathbf{\vec x}}{h}||^2)}- \mathbf{\vec x}$$ 。

7. `Mean-Shift` 算法优点：
   * 簇的数量由算法自动确定，无需人工指定。
   * 基于密度定义，能够对抗噪音。
   * 可以处理任意形状和大小的簇。
   * 没有局部极小值点，因此当给定带宽参数  时，其聚类结果就是唯一的。
8. `Mean_Shift` 算法缺点：
   * 无法控制簇的数量。
   * 无法区分有意义的簇和无意义的簇。如：在`Mean_Shift` 算法中，异常点也会形成它们自己的簇。


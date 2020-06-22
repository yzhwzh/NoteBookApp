# 层次聚类

 层次聚类`hierarchical clustering` 试图在不同层次上对数据集进行划分，从而形成树形的聚类结构。

## 1 AGNES 算法

1. `AGglomerative NESting：AGNES`是一种常用的采用自底向上聚合策略的层次聚类算法。
2. `AGNES`首先将数据集中的每个样本看作一个初始的聚类簇，然后在算法运行的每一步中，找出距离最近的两个聚类簇进行合并。

   合并过程不断重复，直到达到预设的聚类簇的个数。

3. 这里的关键在于：如何计算聚类簇之间的距离？

   由于每个簇就是一个集合，因此只需要采用关于集合的某个距离即可。给定聚类簇 $$\mathbb C_i,\mathbb C_j$$ ， 有三种距离：

   * 最小距离 ： $$d_{min}(\mathbb C_i,\mathbb C_j)=\min_{\mathbf{\vec x}_i \in \mathbb C_i,\mathbf{\vec x}_j \in \mathbb C_j}distance(\mathbf{\vec x}_i,\mathbf{\vec x}_j)$$  。

     最小距离由两个簇的最近样本决定。

   * 最大距离 ： $$d_{max}(\mathbb C_i,\mathbb C_j)=\max_{\mathbf{\vec x}_i \in \mathbb C_i,\mathbf{\vec x}_j \in \mathbb C_j}distance(\mathbf{\vec x}_i,\mathbf{\vec x}_j)$$ 。

     最大距离由两个簇的最远样本决定。

   * 平均距离： $$d_{avg}(\mathbb C_i,\mathbb C_j)=\frac{1}{|\mathbb C_i||\mathbb C_j|}\sum_{\mathbf{\vec x}_i \in \mathbb C_i}\sum_{\mathbf{\vec x}_j \in \mathbb C_j}distance(\mathbf{\vec x}_i,\mathbf{\vec x}_j)$$ 。

     平均距离由两个簇的所有样本决定。

4. `AGNES` 算法可以采取上述任意一种距离：
   * 当`AGNES`算法的聚类簇距离采用 $$d_{min}$$ 时，称作单链接`single-linkage`算法。
   * 当`AGNES`算法的聚类簇距离采用 $$d_{max}$$ 时，称作全链接`complete-linkage`算法。
   * 当`AGNES`算法的聚类簇距离采用  $$d_{avg}$$ 时，称作均链接`average-linkage`算法 。
5. `AGNES`算法：
   * 输入：
     * 数据集 $$\mathbb D=\{\mathbf{\vec x}_1,\mathbf{\vec x}_2,\cdots,\mathbf{\vec x}_N\}$$ 
     * 聚类簇距离度量函数 $$d(\cdot,\cdot)$$ 
     * 聚类簇数量 $$K$$ 
   * 输出：簇划分 $$\mathcal C=\{\mathbb C_1,\mathbb C_2,\cdots,\mathbb C_K\}$$ 
   * 算法步骤：
     * 初始化：将每个样本都作为一个簇 $$\mathbb C_i=\{\mathbf{\vec x}_i\} ,i=1,2,\cdots,N$$ 
     * 迭代，终止条件为聚类簇的数量为 $$K$$ 。迭代过程为：

       计算聚类簇之间的距离，找出距离最近的两个簇，将这两个簇合并。

       > 每进行一次迭代，聚类簇的数量就减少一些。
6. `AGNES` 算法的优点：
   * 距离容易定义，使用限制较少。
   * 可以发现聚类的层次关系。
7. `AGNES` 算法的缺点：
   * 计算复杂度较高。
   * 算法容易聚成链状。

## 4.2 BIRCH 算法

1. `BIRCH:Balanced Iterative Reducing and Clustering Using Hierarchies` 算法通过聚类特征树`CF Tree:Clustering Feature True`来执行层次聚类，适合于样本量较大、聚类类别数较大的场景。

**4.2.1 聚类特征**

1. 聚类特征`CF`：每个`CF` 都是刻画一个簇的特征的三元组： $$CF=(\text{num},\overrightarrow\Sigma_l,\Sigma_s)$$  。其中：
   * $$num$$ ：表示簇内样本数量的数量。
   * $$\overrightarrow\Sigma_l$$ ：表示簇内样本的线性求和： $$\overrightarrow\Sigma_l=\sum_{\mathbf{\vec x}_i\in \mathbb S} \mathbf{\vec x}_i$$ ，其中 $$S$$ 为该`CF` 对应的簇。
   * $$\Sigma_s$$ ：表示簇内样本的长度的平方和。 $$\Sigma_s=\sum_{\mathbf{\vec x}_i\in \mathbb S} ||\mathbf{\vec x}_i||^2=\sum_{\mathbf{\vec x}_i\in \mathbb S} \mathbf{\vec x}_i^T\mathbf{\vec x}_i$$ ，其中 $$S$$ 为该`CF` 对应的簇。
2. 根据`CF` 的定义可知：如果`CF1` 和 `CF2` 分别表示两个不相交的簇的特征，如果将这两个簇合并成一个大簇，则大簇的特征为： $$CF_{merge} = CF_1+CF_2$$ 。

   即：`CF` 满足可加性。

3. 给定聚类特征`CF`，则可以统计出簇的一些统计量：
   * 簇心： $$\bar{\mathbf{\vec x}}=\frac{\vec\Sigma_l}{\text{num}}$$ 。
   * 簇内数据点到簇心的平均距离（也称作簇的半径）： $$\rho=\sqrt{\frac{\text{num}\times \Sigma_s-||\vec\Sigma_l||^2}{\text{num}}}$$ 。
   * 簇内两两数据点之间的平均距离（也称作簇的直径）： $$\delta=\sqrt{\frac{2\times\text{num}\times \Sigma_s-2||\vec\Sigma_l||^2}{\text{num}\times \text{num-1}}}$$ 。
4. 给定两个不相交的簇，其特征分别为 $$CF_1=(\text{num}_1,\overrightarrow\Sigma_{l,1},\Sigma_{s,1})$$ 和 $$CF_2=(\text{num}_2,\overrightarrow\Sigma_{l,2},\Sigma_{s,2})$$ 。

   假设合并之后的簇为 $$CF_3=(\text{num}_3,\overrightarrow\Sigma_{l,3},\Sigma_{s,3})$$ ，其中 $$\text{num}_3=\text{num}_1+\text{num}_2$$ ， $$\overrightarrow\Sigma_{l,3}=\overrightarrow\Sigma_{l,1}+\overrightarrow\Sigma_{l,2}$$ ， $$\Sigma_{s,3}=\Sigma_{s,1}+\Sigma_{s,2}$$ 。

   可以通过下列的距离来度量 `CF1` 和 `CF2` 的相异性：

   * 簇心欧氏距离`centroid Euclidian distance`： $$d_0=\sqrt{||\bar{\mathbf{\vec x}}_1-||\bar{\mathbf{\vec x}}_2||_2^2}$$ ，其中 $$\bar{\mathbf{\vec x}}_1,\bar{\mathbf{\vec x}}_2$$ 分别为各自的簇心。
   * 簇心曼哈顿距离`centroid Manhattan distance`： $$d_1=||\bar{\mathbf{\vec x}}_1-||\bar{\mathbf{\vec x}}_2||_1$$ 。
   * 簇连通平均距离`average inter-cluster distance`： $$d_2=\sqrt{\frac{\sum_{\mathbf{\vec x}_i \in \mathbb S_1}\sum_{\mathbf{\vec x}_j \in \mathbb S_2}||\mathbf{\vec x}_i -\mathbf{\vec x}_j||_2^2}{\text{num}_1\times \text{num}_2}} = \sqrt{\frac{\Sigma_{s,1}}{\text{num}_1}+\frac{\Sigma_{s,2}}{\text{num}_2}-2 \frac{\vec\Sigma_{l,1}^T\vec\Sigma_{l,2}}{\text{num}_1\times\text{num}_2 }}$$ 
   * 全连通平均距离`average intra-cluster distance`： $$d_3=\sqrt{\frac{\sum_{\mathbf{\vec x}_i \in \mathbb S_3}\sum_{\mathbf{\vec x}_j \in \mathbb S_3}||\mathbf{\vec x}_i -\mathbf{\vec x}_j||_2^2}{(\text{num}_1+\text{num}_2)\times (\text{num}_1+\text{num}_2-1)}} \\ = \sqrt{\frac{2(\text{num}_1+\text{num}_2)(\Sigma_{s,1}+\Sigma_{s,2})-2||\vec\Sigma_{l,1}-\vec\Sigma_{l,2}||_2^2}{(\text{num}_1+\text{num}_2)\times (\text{num}_1+\text{num}_2-1)}}$$ 
   * 方差恶化距离`variance incress distance`： $$d_4=\rho_3-\rho_1-\rho_2$$ 。

**4.2.2 CF 树**

1. `CF`树的结构类似于平衡`B+`树 。树由三种结点构成：根结点、中间结点、叶结点。

   * 根结点、中间结点：由若干个聚类特征`CF` ，以及这些`CF` 指向子结点的指针组成。
   * 叶结点：由若干个聚类特征`CF` 组成。
     * 叶结点没有子结点，因此`CF` 没有指向子结点的指针。
     * 所有的叶结点通过双向链表连接起来。
     * 在`BIRCH` 算法结束时，叶结点的每个`CF` 对应的样本集就对应了一个簇。

   ![CF\_Tree](http://www.huaxiaozhuan.com/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0/imgs/clustering/CF_Tree.png)

2. `CF` 树有三个关键参数：
   * 枝平衡因子  $$\beta$$ ：非叶结点中，最多不能包含超过 $$\beta$$ 个 `CF` 。
   * 叶平衡因子 $$\lambda$$ ：叶结点中，最多不能包含超过 $$\lambda$$ 个 `CF` 。
   * 空间阈值 $$\tau$$ ：叶结点中，每个`CF` 对应的子簇的大小（通过簇半径  $$\rho$$ 来描述）不能超过 $$\tau$$ 。
3. 由于`CF` 的可加性，所以`CF` 树中，每个父结点的`CF` 等于它所有子结点的所有`CF` 之和。
4. `CF` 树的生成算法：
   * 输入：
     * 样本集 $$\mathbb D=\{\mathbf{\vec x}_1,\cdots,\mathbf{\vec x}_N\}$$ 
     * 枝平衡因子 $$\beta$$ 
     * 叶平衡因子 $$\lambda$$ 
     * 空间阈值 $$\tau$$ 
   * 输出：`CF` 树
   * 算法步骤：
     * 初始化：`CF` 树的根结点为空。
     * 随机从样本集 $$\mathbb D$$ 中选出一个样本，放入一个新的`CF` 中，并将该`CF` 放入到根结点中。
     * 遍历 $$\mathbb D$$ 中的样本，并向`CF` 树中插入。迭代停止条件为：样本集 $$\mathbb D$$ 中所有样本都插入到`CF` 树中。

       迭代过程如下：

       * 随机从样本集 $$\mathbb D$$ 中选出一个样本 $$\mathbf{\vec x}_i$$ ，从根结点向下寻找与 $$\mathbf{\vec x}_i$$ 距离最近的叶结点 $$\text{leaf}_j$$ ，和 $$\text{leaf}_j$$ 里与 $$\mathbf{\vec x}_i$$ 最近的  $$CF_k$$ 。
       * 如果 $$\mathbf{\vec x}_i$$ 加入到 $$CF_k$$ 对应的簇中之后，该簇的簇半径 $$\rho \le \tau$$ ，则将 $$\mathbf{\vec x}_i$$ 加入到  $$CF_k$$ 对应的簇中，并更新路径上的所有`CF` 。本次插入结束。
       * 否则，创建一个新的`CF`，将 $$\mathbf{\vec x}_i$$ 放入该`CF` 中，并将该`CF` 添加到叶结点 $$\text{leaf}_j$$ 中。

         如果  $$\text{leaf}_j$$ 的`CF` 数量小于 $$\lambda $$ ，则更新路径上的所有`CF` 。本次插入结束。

       * 否则，将叶结点 $$\text{leaf}_j$$ 分裂为两个新的叶结点 $$\text{leaf}_{j,1},leaf_{j,2}$$  。分裂方式为：
         * 选择叶结点 $$\text{leaf}_j$$ 中距离最远的两个`CF`，分别作为  $$\text{leaf}_{j,1},leaf_{j,2}$$ 中的首个`CF` 。
         * 将叶结点 $$\text{leaf}_j$$ 中剩下的`CF` 按照距离这两个`CF` 的远近，分别放置到 $$\text{leaf}_{j,1},leaf_{j,2}$$ 中。
       * 依次向上检查父结点是否也需要分裂。如果需要，则按照和叶子结点分裂方式相同。

**4.2.3 BIRCH 算法**

1. `BIRCH` 算法的主要步骤是建立`CF` 树，除此之外还涉及到`CF` 树的瘦身、离群点的处理。
2. `BIRCH` 需要对`CF` 树瘦身，有两个原因：
   * 将数据点插入到`CF` 树的过程中，用于存储`CF` 树结点及其相关信息的内存有限，导致部分数据点生长形成的`CF` 树占满了内存。因此需要对`CF` 树瘦身，从而使得剩下的数据点也能插入到`CF` 树中。
   * `CF` 树生长完毕后，如果叶结点中的`CF` 对应的簇太小，则会影响后续聚类的速度和质量。
3. `BIRCH` 瘦身是在将 $$\tau$$ 增加的过程。算法会在内存中同时存放旧树 $$\mathcal T$$ 和新树 $$\mathcal T^\prime$$ ，初始时刻 $$\mathcal T^\prime$$ 为空。
   * 算法同时处理 $$\mathcal T$$ 和 $$\mathcal T^\prime$$ ，将  $$\mathcal T$$ 中的 `CF` 迁移到  $$\mathcal T^\prime$$ 中。
   * 在完成所有的`CF` 迁移之后， $$\mathcal T$$ 为空， $$\mathcal T^\prime$$  就是瘦身后的 `CF` 树。
4. `BIRCH` 离群点的处理：
   * 在对`CF` 瘦身之后，搜索所有叶结点中的所有子簇，寻找那些稀疏子簇，并将稀疏子簇放入待定区。

     稀疏子簇：簇内数据点的数量远远少于所有子簇的平均数据点的那些子簇。

     > 将稀疏子簇放入待定区时，需要同步更新`CF` 树上相关路径及结点。

   * 当 $$D$$ 中所有数据点都被插入之后，扫描待定区中的所有数据点（这些数据点就是候选的离群点），并尝试将其插入到`CF` 树中。

     如果数据点无法插入到`CF` 树中，则可以确定为真正的离群点。
5. `BIRCH` 算法：
   * 输入：
     * 样本集 $$\mathbb D=\{\mathbf{\vec x}_1,\cdots,\mathbf{\vec x}_N\}$$ 
     * 枝平衡因子 $$\beta$$ 
     * 叶平衡因子 $$\lambda$$ 
     * 空间阈值 $$\tau$$ 
   * 输出：`CF` 树
   * 算法步骤：
     * 建立 `CF` 树。
     * （可选）对`CF` 树瘦身、去除离群点，以及合并距离非常近的`CF` 。
     * （可选）利用其它的一些聚类算法（如：`k-means`）对`CF`树的所有叶结点中的`CF` 进行聚类，得到新的`CF` 树。

       这是为了消除由于样本读入顺序不同导致产生不合理的树结构。

       > 这一步是对 `CF` 结构进行聚类。由于每个`CF` 对应一组样本，因此对`CF` 聚类就是对  进行聚类。

     * （可选）将上一步得到的、新的`CF` 树的叶结点中每个簇的中心点作为簇心，对所有样本按照它距这些中心点的距离远近进行聚类。

       这是对上一步的结果进行精修。
6. `BIRCH` 算法优点：
   * 节省内存。所有样本都存放在磁盘上，内存中仅仅存放`CF` 结构。
   * 计算速度快。只需要扫描一遍就可以建立`CF` 树。
   * 可以识别噪声点。
7. `BIRCH` 算法缺点：
   * 结果依赖于数据点的插入顺序。原本属于同一个簇的两个点可能由于插入顺序相差很远，从而导致分配到不同的簇中。

     甚至同一个点在不同时刻插入，也会被分配到不同的簇中。

   * 对非球状的簇聚类效果不好。这是因为簇直径  和簇间距离的计算方法导致。
   * 每个结点只能包含规定数目的子结点，最后聚类的簇可能和真实的簇差距很大。
8. `BIRCH` 可以不用指定聚类的类别数  。
   * 如果不指定 $$K$$ ，则最终叶结点中`CF` 的数量就是 $$K$$ 。
   * 如果指定 $$K$$ ，则需要将叶结点按照距离远近进行合并，直到叶结点中`CF` 数量等于  $$K$$ 。


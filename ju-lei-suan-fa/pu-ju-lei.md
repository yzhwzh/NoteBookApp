# 谱聚类

1. 谱聚类`spectral clustering` 是一种基于图论的聚类方法。
2. 谱聚类的主要思想是：基于数据集 $$\mathbb D=\{\mathbf{\vec x}_1,\cdots,\mathbf{\vec x}_N\}$$ 来构建图 $$\mathcal G=(\mathbb V,\mathbb E)$$ ，其中：

   * 顶点 $$\mathbb V$$ ：由数据集中的数据点组成： $$\mathbb V=\{1,2,\cdots,N\}$$ 。
   * 边 $$\mathbb E$$ ：任意一对顶点之间存在边。

     距离越近的一对顶点，边的权重越高；距离越远的一对顶点，边的权重越低。

   通过对图 $$\mathcal G$$ 进行切割，使得切割之后：不同子图之间的边的权重尽可能的低、各子图内的边的权重尽可能的高。这样就完成了聚类。

   ![spectral\_cluster](http://www.huaxiaozhuan.com/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0/imgs/clustering/spectral_cluster1.png)

#### 5.1 邻接矩阵

1. 在图 $$\mathcal G=(\mathbb V,\mathbb E)$$ 中，定义权重 $$w_{i,j}$$ 为顶点 $$i$$ 和 $$j$$ 之间的权重，其中 $$i,j \in \mathbb V$$ 。

   定义 $$\mathbf W=(w_{i,j})_{N\times N}$$ 为邻接矩阵： $$\mathbf W=\begin{bmatrix} w_{1,1}&w_{1,2}&\cdots&w_{1,N}\\ w_{2,1}&w_{2,2}&\cdots&w_{2,N}\\ \vdots&\vdots&\ddots&\vdots\\ w_{N,1}&w_{N,2}&\cdots&w_{N,N} \end{bmatrix}$$ 

   由于 $$\mathcal G$$ 为无向图，因此 $$w_{i,j}=w_{j,i}$$ 。即： $$\mathbf W=\mathbf W^T$$  。

   * 对图中顶点 $$i$$ ，定义它的度 $$d_i$$ 为：所有与顶点 $$i$$ 相连的边的权重之和： $$d_i=\sum_{j=1}^N w_{i,j}$$ 。

     定义度矩阵 $$D$$ 为一个对角矩阵，其中对角线分别为各顶点的度： $$\mathbf D=\begin{bmatrix} d_1&0&\cdots&0\\ 0&d_2&\cdots&0\\ \vdots&\vdots&\ddots&\vdots\\ 0&0&\cdots&d_N\\ \end{bmatrix}$$ 

   * 对于顶点集合 $$\mathbb V$$ 的一个子集 $$\mathbb A \in \mathbb V$$ ，定义 $$|\mathbb A|$$ 为子集 $$\mathbb A$$ __中点的个数；定义  $$vol(\mathbb A)=\sum_{i\in \mathbb A}d_i$$ ，为子集 $$\mathbb A$$ 中所有点的度之和。

2. 事实上在谱聚类中，通常只给定数据集 $$\mathbb D=\{\mathbf{\vec x}_1,\cdots,\mathbf{\vec x}_N\}$$ ，因此需要计算出邻接矩阵 $$W$$ 。
   * 基本思想是：距离较近的一对点（即相似都较高），边的权重较高；距离较远的一对点（即相似度较低），边的权重较低。
   * 基本方法是：首先构建相似度矩阵 $$\mathbf S=(s_{i,j})_{N\times N}$$ ，然后使用 $$\epsilon$$ -近邻法、 $$K$$ 近邻法、或者全连接法。 $$\mathbf S=\begin{bmatrix} s_{1,1}&s_{1,2}&\cdots&s_{1,N}\\ s_{2,1}&s_{2,2}&\cdots&s_{2,N}\\ \vdots&\vdots&\ddots&\vdots\\ s_{N,1}&s_{N,2}&\cdots&s_{N,N} \end{bmatrix}$$ 
     * 通常相似度采用高斯核： $$s_{i,j}=\exp\left(-\frac{||\mathbf{\vec x}_i-\mathbf{\vec x}_j||_2^2}{2\sigma^2}\right)$$ 。此时有 $$s_{i,j}=s_{j,i}$$ 。即： $$\mathbf S=\mathbf S^T$$  。
     * 也可以选择不同的核函数，如：多项式核函数、高斯核函数、`sigmoid` 核函数。
3. $$\epsilon$$ -近邻法：设置一个距离阈值 $$\epsilon$$ ，定义邻接矩阵 $$W$$  为： $$w_{i,j}=\begin{cases} 0, & s_{i,j} \gt \varepsilon\\ \varepsilon , & s_{i,j} \le \varepsilon  \end{cases}$$ 

   即：一对相似度小于 $$\epsilon$$ 的点，边的权重为 $$\epsilon$$ ；否则边的权重为 0 。

   $$\epsilon$$ -**近邻法得到的权重要么是 0 ，要么是** $$\epsilon$$ **，权重度量很不精确，因此实际应用较少**。

4.  $$K$$ 近邻法：利用 `KNN` 算法选择每个样本最近的  $$K$$ 个点作为近邻，其它点与当前点之间的边的权重为 0 。

   这种做法会导致邻接矩阵 $$W$$ 非对称，因为当 $$\mathbf{\vec x}_j$$ 是 $$\mathbf{\vec x}_i$$ 的  $$K$$ 近邻时，  $$\mathbf{\vec x}_i$$ 不一定是 $$\mathbf{\vec x}_j$$ 的 $$K$$  近邻。

   为了解决对称性问题，有两种做法：

   * 只要一个点在另一个点的 $$K$$ 近邻中，则认为是近邻。即：取并集 $$w_{i,j}=w_{j,i}=\begin{cases} 0,&\mathbf{\vec x}_i \notin KNN(\mathbf{\vec x}_j) \;\text{and}\;\mathbf{\vec x}_j \notin KNN(\mathbf{\vec x}_i) \\ s_{i,j},&\mathbf{\vec x}_i \in KNN(\mathbf{\vec x}_j) \;\text{or}\;\mathbf{\vec x}_j \in KNN(\mathbf{\vec x}_i)  \end{cases}$$ 。
   * 只有两个点互为对方的 $$K$$ 近邻中，则认为是近邻。即：取交集 $$w_{i,j}=w_{j,i}=\begin{cases} 0,&\mathbf{\vec x}_i \notin KNN(\mathbf{\vec x}_j) \;\text{or}\;\mathbf{\vec x}_j \notin KNN(\mathbf{\vec x}_i) \\ s_{i,j},&\mathbf{\vec x}_i \in KNN(\mathbf{\vec x}_j) \;\text{and}\;\mathbf{\vec x}_j \in KNN(\mathbf{\vec x}_i)  \end{cases}$$ 。

5. 全连接法：所有点之间的权重都大于 0 ： $$w_{i,j}= s_{i,j}$$ 。

#### 5.2 拉普拉斯矩阵

1. 定义拉普拉斯矩阵 $$\mathbf L=\mathbf D-\mathbf W$$ ，其中 $$D$$ 为度矩阵、 $$W$$ 为邻接矩阵。
2. 拉普拉斯矩阵 $$L$$ 的性质：
   * $$L$$ 是对称矩阵，即  $$\mathbf L=\mathbf L^T$$ 。这是因为 $$\mathbf D,\mathbf W$$  都是对称矩阵。
   * 因为 $$L$$ 是实对称矩阵，因此它的特征值都是实数。
   * 对任意向量 $$\mathbf{\vec f}=(f_1,f_2,\cdots,f_N)^T$$ ，有： $$\mathbf{\vec f}^T\mathbf L\mathbf{\vec f}=\frac 12 \sum_{i=1}^N\sum_{j=1}^N w_{i,j}(f_i-f_j)^2$$ 。
   *  $$L$$ 是半正定的，且对应的 $$N$$ 个特征值都大于等于0，且最小的特征值为 0。

     设其  $$N$$ 个实特征值从小到大为 $$\lambda_1,\cdots,\lambda_N$$ ，即： $$0=\lambda_1\le\lambda_2\le\cdots\le\lambda_N$$ 。

#### 5.3 谱聚类算法

1. 给定无向图 $$\mathcal G=(\mathbb V,\mathbb E)$$ ，设子图的点的集合 $$\mathbb A$$ 和子图的点的集合 $$\mathbb B$$ 都是  $$\mathbb V$$ 的子集，且  $$\mathbb A\bigcap \mathbb B=\phi$$ 。

   定义 $$\mathbb A$$ 和  $$\mathbb B$$ 之间的切图权重为： $$W(\mathbb A,\mathbb B)=\sum_{i\in \mathbb A,j\in \mathbb B}w_{i,j}$$ 。

   即：所有连接 $$\mathbb A$$ 和  $$\mathbb B$$ 之间的边的权重。

2. 对于无向图  $$\mathcal G=(\mathbb V,\mathbb E)$$ ，假设将它切分为 $$k$$ 个子图：每个子图的点的集合为 $$\mathbb A_1,\cdots,\mathbb A_k$$ ，满足 $$\mathbb A_i\bigcap\mathbb A_j=\phi,i\ne j$$ 且  $$\mathbb A_1\bigcup\cdots\bigcup\mathbb A_k=\mathbb V$$ 。

   定义切图`cut` 为： $$cut(\mathbb A_1,\cdots,\mathbb A_k)=  \sum_{i=1}^kW(\mathbb A_i,\bar{\mathbb A}_i)$$ ，其中 $$\bar{\mathbb A}_i$$ 为 $$\mathbb A_i$$ 的补集。

**5.3.1 最小切图**

1. 引入指示向量 $$\mathbf{\vec q}_j=(q_{j,1},\cdots,q_{j,N})^T,j=1,2,\cdots,k$$ ，定义： $$q_{j,i}=\begin{cases} 0,& i \not \in \mathbb A_j\\ 1,& i \in\mathbb A_j \end{cases}$$ 

   则有： $$\mathbf{\vec q}_j^T\mathbf L\mathbf{\vec q}_j=\frac 12 \sum_{m=1}^N\sum_{n=1}^N w_{m,n}(q_{j,m}-q_{j,n})^2\\ =\frac 12 \sum_{m\in \mathbb A_j}\sum_{n\in \mathbb A_j} w_{m,n}(1-1)^2+ \frac 12  \sum_{m\not\in \mathbb A_j}\sum_{n\not\in \mathbb A_j} w_{m,n}(0-0)^2+\\ \frac 12 \sum_{m\in \mathbb A_j}\sum_{n\not\in \mathbb A_j} w_{m,n}(1-0)^2+\frac 12  \sum_{m\not\in \mathbb A_j}\sum_{n\in \mathbb A_j}  w_{m,n}(0-1)^2\\ =\frac 12\left(\sum_{m\in \mathbb A_j}\sum_{n\not\in \mathbb A_j} w_{m,n}+\sum_{m\not\in \mathbb A_j}\sum_{n\in \mathbb A_j}  w_{m,n}\right)\\ =\frac 12(cut(\mathbb A_j,\bar{\mathbb A}_j)+cut(\bar{\mathbb A}_j,\mathbb A_j))=cut(\mathbb A_j,\bar{\mathbb A}_j)$$ 

   因此  $$cut(\mathbb A_1,\cdots,\mathbb A_k)=\sum_{j=1}^k\mathbf{\vec q}_j^T\mathbf L\mathbf{\vec q}_j=tr(\mathbf Q^T\mathbf L\mathbf Q)$$ 。其中 $$\mathbf Q=(\mathbf{\vec q}_1,\cdots,\mathbf{\vec q}_k)$$ ， $$tr(\cdot)$$ 为矩阵的迹。

   考虑到顶点 $$i$$ 有且仅位于一个子图中，则有约束条件： $$q_{j,m}\in \{0,1\},\quad \mathbf{\vec q}_i\cdot\mathbf{\vec q}_j=\begin{cases} 0,&i\ne j\\ |\mathbb A|_j,&i = j \end{cases}$$ 

2. 最小切图算法： $$cut(\mathbb A_1,\cdots,\mathbb A_k)$$  最小的切分。即求解： $$\min_{\mathbf Q} tr(\mathbf Q^T\mathbf L\mathbf Q)\\ s.t. q_{j,m}\in \{0,1\},\quad \mathbf{\vec q}_i\cdot\mathbf{\vec q}_j=\begin{cases} 0,&i\ne j\\ |\mathbb A|_j,&i = j \end{cases}$$ 
3. 最小切图切分使得不同子图之间的边的权重尽可能的低，但是容易产生分割出只包含几个顶点的较小子图的歪斜分割现象。

   ![spectral\_cluster](http://www.huaxiaozhuan.com/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0/imgs/clustering/spectral_cluster2.png)

**5.3.2 RatioCut 算法**

1. `RatioCut` 切图不仅考虑最小化 $$cut(\mathbb A_1,\cdots,\mathbb A_k)$$ ，它还考虑最大化每个子图的点的个数。即： $$RatioCut(\mathbb A_1,\cdots,\mathbb A_k)=  \sum_{i=1}^k\frac{W(\mathbb A_i,\bar{\mathbb A}_i)}{|\mathbb A_i|}$$ 
   * 最小化 $$cut(\mathbb A_1,\cdots,\mathbb A_k)$$ ：使得不同子图之间的边的权重尽可能的低。
   * 最大化每个子图的点的个数：使得各子图尽可能的大。
2. 引入指示向量 $$\mathbf{\vec h}_j=(h_{j,1},\cdots,h_{j,N})^T,j=1,2,\cdots,k$$ ，定义： $$h_{j,i}=\begin{cases} 0,& i \not \in \mathbb A_j\\ \frac{1}{\sqrt{|\mathbb A_j|}},& i \in\mathbb A_j \end{cases}$$ 

   则有： $$\mathbf{\vec h}_j^T\mathbf L\mathbf{\vec h}_j=\frac 12 \sum_{m=1}^N\sum_{n=1}^N w_{m,n}(h_{j,m}-h_{j,n})^2\\ =\frac 12 \sum_{m\in \mathbb A_j}\sum_{n\not\in \mathbb A_j} w_{m,n}(\frac{1}{\sqrt{|\mathbb A_j|}}-0)^2+\frac 12  \sum_{m\not\in \mathbb A_j}\sum_{n\in \mathbb A_j}  w_{m,n}(0-\frac{1}{\sqrt{|\mathbb A_j|}})^2\\ =\frac 12\left(\sum_{m\in \mathbb A_j}\sum_{n\not\in \mathbb A_j} \frac{w_{m,n}}{|\mathbb A_j|}+\sum_{m\not\in \mathbb A_j}\sum_{n\in \mathbb A_j} \frac {w_{m,n}}{|\mathbb A_j|}\right)\\ =\frac 12\times\frac{1}{|\mathbb A_j|}(cut(\mathbb A_j,\bar{\mathbb A}_j)+cut(\bar{\mathbb A}_j,\mathbb A_j))=RatioCut(\mathbb A_j,\bar{\mathbb A}_j)$$ 

   因此 $$RatioCut(\mathbb A_1,\cdots,\mathbb A_k)=\sum_{j=1}^k\mathbf{\vec h}_j^T\mathbf L\mathbf{\vec h}_j=tr(\mathbf H^T\mathbf L\mathbf H)$$ 。其中  $$\mathbf H=(\mathbf{\vec h}_1,\cdots,\mathbf{\vec h}_k)$$ ， 为矩阵的迹。

   考虑到顶点 $$i$$ 有且仅位于一个子图中，则有约束条件： $$\mathbf{\vec h}_i\cdot\mathbf{\vec h}_j=\begin{cases} 0,&i\ne j\\ 1,&i = j \end{cases}$$ 

3. `RatioCut`算法： $$RatioCut(\mathbb A_1,\cdots,\mathbb A_k)$$ 最小的切分。即求解： $$\min_{\mathbf H} tr(\mathbf H^T\mathbf L\mathbf H)\\   s.t. \mathbf H^T\mathbf H=\mathbf I$$ 

   因此只需要求解 $$L$$ 最小的 $$k$$ 个特征值，求得对应的 $$k$$ 个特征向量组成 $$H$$ 。

   通常在求解得到  $$H$$ 之后，还需要对行进行标准化： $$h_{i,j}^*=\frac{h_{i,j}}{\sqrt{\sum_{t=1}^kh_{i,t}^2}}$$ 

4. 事实上这样解得的 $$H$$ 不能完全满足指示向量的定义。因此在得到  $$H$$ 之后，还需要对每一行进行一次传统的聚类（如：`k-means` 聚类）。
5. `RatioCut` 算法：
   * 输入：
     * 数据集 $$\mathbb D=\{\mathbf{\vec x}_1,\cdots,\mathbf{\vec x}_N\}$$ 
     * 降维的维度 $$k_1$$ 
     * 二次聚类算法
     * 二次聚类的维度 $$k_2$$ 
   * 输出：聚类簇 $$\mathcal C=\{\mathbb C_1,\cdots,\mathbb C_{k_2}\}$$ 
   * 算法步骤：
     * 根据 $$\mathbb D$$ 构建相似度矩阵 $$S$$ 。
     * 根据相似度矩阵构建邻接矩阵 $$W$$ 、度矩阵 $$D$$ ，计算拉普拉斯矩阵  $$L=D-W$$ 。
     * 计算 $$L$$ 最小的 $$k_1$$ 个特征值，以及对应的特征向量 $$\mathbf{\vec v}_1,\cdots,\mathbf{\vec v}_{k_1}$$ ，构建矩阵  $$\mathbf H=(\mathbf{\vec v}_1,\cdots,\mathbf{\vec v}_{k_1})$$ 。
     * 对 $$H$$ 按照行进行标准化： $$h_{i,j}^*=\frac{h_{i,j}}{\sqrt{\sum_{t=1}^kh_{i,t}^2}}$$ ，得到  $$\mathbf H^*$$ 。
     * 将 $$\mathbf H^*$$ 中每一行作为一个 $$k_1$$ 维的样本，一共 $$N$$ 个样本，利用二次聚类算法来聚类，二次聚类的维度为  $$k_2$$ 。

       最终得到簇划分 $$\mathcal C=\{\mathbb C_1,\cdots,\mathbb C_{k_2}\}$$ 。

**5.3.3 Ncut 算法**

1. `Ncut` 切图不仅考虑最小化 $$cut(\mathbb A_1,\cdots,\mathbb A_k)$$ ，它还考虑最大化每个子图的边的权重。即： $$Ncut(\mathbb A_1,\cdots,\mathbb A_k)=  \sum_{i=1}^k\frac{W(\mathbb A_i,\bar{\mathbb A}_i)}{vol(\mathbb A_i)}$$ 
   * 最小化 $$cut(\mathbb A_1,\cdots,\mathbb A_k)$$ ：使得不同子图之间的边的权重尽可能的低。
   * 最大化每个子图的边的权重：使得各子图内的边的权重尽可能的高。
2. 引入指示向量 $$\mathbf{\vec h}_j=(h_{j,1},\cdots,h_{j,N})^T,j=1,2,\cdots,k$$ ，定义： $$h_{j,i}=\begin{cases} 0,& i \not \in \mathbb A_j\\ \frac{1}{\sqrt{vol(\mathbb A_j)}},& i \in\mathbb A_j \end{cases}$$ 

   则有： $$\mathbf{\vec h}_j^T\mathbf L\mathbf{\vec h}_j=\frac 12 \sum_{m=1}^N\sum_{n=1}^N w_{m,n}(h_{j,m}-h_{j,n})^2\\ =\frac 12 \sum_{m\in \mathbb A_j}\sum_{n\not\in \mathbb A_j} w_{m,n}(\frac{1}{\sqrt{vol(\mathbb A_j)}}-0)^2+\frac 12  \sum_{m\not\in \mathbb A_j}\sum_{n\in \mathbb A_j}  w_{m,n}(0-\frac{1}{\sqrt{vol(\mathbb A_j)}})^2\\ =\frac 12\left(\sum_{m\in \mathbb A_j}\sum_{n\not\in \mathbb A_j} \frac{w_{m,n}}{vol(\mathbb A_j)}+\sum_{m\not\in \mathbb A_j}\sum_{n\in \mathbb A_j} \frac {w_{m,n}}{vol(\mathbb A_j)}\right)\\ =\frac 12\times\frac{1}{vol(\mathbb A_j)}(cut(\mathbb A_j,\bar{\mathbb A}_j)+cut(\bar{\mathbb A}_j,\mathbb A_j))=Ncut(\mathbb A_j,\bar{\mathbb A}_j)$$ 

   因此 $$Ncut(\mathbb A_1,\cdots,\mathbb A_k)=\sum_{j=1}^k\mathbf{\vec h}_j^T\mathbf L\mathbf{\vec h}_j=tr(\mathbf H^T\mathbf L\mathbf H)$$ 。其中 $$\mathbf H=(\mathbf{\vec h}_1,\cdots,\mathbf{\vec h}_k)$$ ， $$tr(\cdot)$$ 为矩阵的迹。

   考虑到顶点 $$i$$ 有且仅位于一个子图中，则有约束条件： $$\mathbf{\vec h}_i\cdot\mathbf{\vec h}_j=\begin{cases} 0,&i\ne j\\ \frac{1}{ vol(\mathbb A_j)},&i = j \end{cases}$$ 

3. `Ncut`算法： $$Ncut(\mathbb A_1,\cdots,\mathbb A_k)$$ 最小的切分。即求解 $$\min_{\mathbf H} tr(\mathbf H^T\mathbf L\mathbf H)\\   s.t. \mathbf H^T\mathbf D\mathbf H=\mathbf I$$ 
4. 令  $$\mathbf H=\mathbf D^{-1/2}\mathbf F$$ ，则有： $$\mathbf H^T\mathbf L\mathbf H=\mathbf F^T\mathbf D^{-1/2}\mathbf L\mathbf D^{-1/2}\mathbf F\\ \mathbf H^T\mathbf D\mathbf H=\mathbf F^T\mathbf F=\mathbf I$$ 

   令  $$\mathbf L^\prime=\mathbf D^{-1/2}\mathbf L\mathbf D^{-1/2}$$ ，则最优化目标变成： $$\min_{\mathbf H} tr(\mathbf F^T\mathbf L^\prime\mathbf F)\\   s.t. \mathbf F^T\mathbf F=\mathbf I$$ 

   因此只需要求解  $$\mathbf L^\prime$$ 最小的  $$k$$ 个特征值，求得对应的 $$k$$ 个特征向量组成  $$F$$ 。然后对行进行标准化： $$f_{i,j}^*=\frac{f_{i,j}}{\sqrt{\sum_{t=1}^kf_{i,t}^2}}$$ 。

   与`RatioCut` 类似，`Ncut` 也需要对 $$F$$ 的每一行进行一次传统的聚类（如：`k-means` 聚类）。

5. 事实上  $$\mathbf D^{-1/2}\mathbf L\mathbf D^{-1/2}$$ 相当于对拉普拉斯矩阵 $$L$$ 进行了一次标准化： $$l_{i,j}^\prime=\frac{l_{i,j}}{d_i\times d_j}$$ 。
6. `Ncut` 算法：
   * 输入：
     * 数据集 $$\mathbb D=\{\mathbf{\vec x}_1,\cdots,\mathbf{\vec x}_N\}$$ 
     * 降维的维度 $$k_1$$ 
     * 二次聚类算法
     * 二次聚类的维度 $$k_2$$ 
   * 输出：聚类簇 $$\mathcal C=\{\mathbb C_1,\cdots,\mathbb C_{k_2}\}$$ 
   * 算法步骤：
     * 根据  $$D$$ 构建相似度矩阵  $$S$$ 。
     * 根据相似度矩阵构建邻接矩阵 $$W$$ 、度矩阵  $$D$$ ，计算拉普拉斯矩阵  $$L=D-W$$ 。
     * 构建标准化之后的拉普拉斯矩阵  $$\mathbf L^\prime= \mathbf D^{-1/2}\mathbf L\mathbf D^{-1/2}$$ 。
     * 计算 $$\mathbf L^\prime$$ 最小的 $$k_1$$ 个特征值，以及对应的特征向量 $$\mathbf{\vec v}_1,\cdots,\mathbf{\vec v}_{k_1}$$ ，构建矩阵  $$\mathbf F=(\mathbf{\vec v}_1,\cdots,\mathbf{\vec v}_{k_1})$$ 。
     * 对 $$F$$ 按照行进行标准化： $$f_{i,j}^*=\frac{f_{i,j}}{\sqrt{\sum_{t=1}^kf_{i,t}^2}}$$ ，得到 $$\mathbf F^*$$ 。
     * 将 $$\mathbf F^*$$ 中每一行作为一个  $$k_1$$ 维的样本，一共 $$N$$ 个样本，利用二次聚类算法来聚类，二次聚类的维度为  $$k_2$$ 。

       最终得到簇划分 $$\mathcal C=\{\mathbb C_1,\cdots,\mathbb C_{k_2}\}$$ 。

**5.4 性质**

1. 谱聚类算法优点：
   * 只需要数据之间的相似度矩阵，因此处理稀疏数据时很有效。
   * 由于使用了降维，因此在处理高维数据聚类时效果较好。
2. 谱聚类算法缺点：
   * 如果最终聚类的维度非常高，则由于降维的幅度不够，则谱聚类的运行速度和最后聚类的效果均不好。
   * 聚类效果依赖于相似度矩阵，不同相似度矩阵得到的最终聚类效果可能不同。


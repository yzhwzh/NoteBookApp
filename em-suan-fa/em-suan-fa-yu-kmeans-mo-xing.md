# EM 算法与 kmeans 模型

1. `kmeans`算法：给定样本集 $$\mathbb D=\{\mathbf{\vec x}_1,\mathbf{\vec x}_2,\cdots,\mathbf{\vec x}_N\}$$ ， 针对聚类所得簇划分 $$\mathcal C=\{\mathbb C_1,\mathbb C_2,\cdots,\mathbb C_K\}$$ ， 最小化平方误差： $$\min_{\mathcal C} \sum_{k=1}^{K}\sum_{\mathbf{\vec x}_i \in \mathbb C_k}||\mathbf{\vec x}_i-\vec \mu_k||_2^{2}$$ 

   其中 $$\vec \mu_k=\frac {1}{|\mathbb C_k|}\sum_{\mathbf{\vec x}_i \in \mathbb C_k}\mathbf{\vec x}_i$$ 是簇 $$\mathbb C_k$$ 的均值向量。

2. 定义观测随机变量为 $$\mathbf {\vec x}$$  ，观测数据为 $$\mathbb D$$ 。定义隐变量为  $$z$$ ，它表示 $$\mathbf {\vec x}$$ 所属的簇的编号。设参数 $$\theta= (\mathbf {\vec\mu}_1,\mathbf {\vec\mu}_2,\cdots,\mathbf {\vec\mu}_K)$$ ， 则考虑如下的生成模型： $$P(\mathbf {\vec x},z\mid\theta) \propto  \begin{cases}\exp(-||\mathbf {\vec x}-\mathbf {\vec \mu}_z||_2^2)\quad &||\mathbf {\vec x}-\mathbf {\vec \mu}_z||_2^2=\min_{1\le k\le K}||\mathbf {\vec x}-\mathbf {\vec \mu}_k||_2^2\\ 0\quad &||\mathbf {\vec x}-\mathbf {\vec \mu}_z||_2^2\gt \min_{1\le k\le K}||\mathbf {\vec x}-\mathbf {\vec \mu}_k||_2^2 \end{cases}$$ 

   其中 $$ \min_{1\le k\le K}||\mathbf {\vec x}-\mathbf {\vec \mu}_k||_2^2$$ 表示距离  $$\mathbf{\vec x}$$ 最近的中心点所在的簇编号。即：

   * 若 $$\mathbf {\vec x}$$ 最近的簇就是 $$\mathbf{\vec \mu}_z$$ 代表的簇，则生成概率为 $$\exp(-||\mathbf {\vec x}-\mathbf {\vec \mu}_z||_2^2)$$ 。
   * 若 $$\mathbf {\vec x}$$ 最近的簇不是 $$\mathbf{\vec \mu}_z$$ 代表的簇，则生成概率等于 0 。

3. 计算后验概率： $$P(z\mid \mathbf{\vec  x},\theta^{<i>})\propto \begin{cases} 1\quad &||\mathbf {\vec x}_i-\mathbf {\vec \mu}_z||_2^2=\min_{1\le k\le K}||\mathbf {\vec x}-\mathbf {\vec \mu}_k^{<i>}||_2^2\\ 0\quad &||\mathbf {\vec x}_i-\mathbf {\vec \mu}_z||_2^2\gt \min_{1\le k\le K}||\mathbf {\vec x}-\mathbf {\vec \mu}_k^{<i>}||_2^2 \end{cases}$$ 

   即：

   * 若 $$\mathbf{\vec x}$$ 最近的簇就是 $$\mathbf{\vec \mu}_z$$  代表的簇，则后验概率为 1 。
   * 若 $$\mathbf{\vec x}$$ 最近的簇不是 $$\mathbf{\vec \mu}_z$$ 代表的簇，则后验概率为 0 。

4. 计算 $$Q$$ 函数： $$Q(\theta,\theta^{<i>})=\sum_{j=1}^N\left(\sum_z P(z\mid \mathbf{\vec  x}=\mathbf{\vec  x}_j;\theta^{<i>})\log P(\mathbf{\vec  x}=\mathbf{\vec  x}_j,z;\theta) \right)$$ 

   设距离 $$\mathbf{\vec x}_j$$  最近的聚类中心为 $$\mathbf{\vec \mu}_{t_j}^{<i>}$$ ，即它属于簇 $$t_j$$ ，则有： $$Q(\theta,\theta^{<i>})=\text{const}-\sum_{j=1}^N ||\mathbf{\vec x}_j-\vec\mu_{t_j}||_2^2$$ 

   则有： $$\theta^{<i+1>}=\arg\max_\theta Q(\theta,\theta^{<i>})=\arg\min_\theta \sum_{j=1}^N ||\mathbf{\vec x}_j-\vec\mu_{t_j}||_2^2$$ 

   定义集合  $$\mathbb I_k=\{j\mid t_j=k\},\quad k=1,2\cdots,K$$ ，它表示属于簇 $$k$$ 的样本的下标集合。则有： $$\sum_{j=1}^N ||\mathbf{\vec x}_j-\vec\mu_{t_j}||_2^2=\sum_{k=1}^K\sum_{j\in \mathbb I_k}  ||\mathbf{\vec x}_j-\vec\mu_k||_2^2$$ 

   则有： $$\theta^{<i+1>}=\arg\min_\theta\sum_{k=1}^K\sum_{j\in \mathbb I_k}  ||\mathbf{\vec x}_j-\vec\mu_k||_2^2$$ 

   这刚好就是 `k-means` 算法的目标：最小化平方误差。

5. 由于求和的每一项都是非负的，则当每一个内层求和 $$\sum_{j\in \mathbb I_k}||\mathbf{\vec x}_j-\mathbf{\vec\mu}_{k}||_2^2$$ 都最小时，总和最小。即： $$\vec\mu^{<i+1>}_k=\arg\min_{\vec\mu_k}\sum_{j\in \mathbb I_k}||\mathbf{\vec x}_j-\mathbf{\vec\mu}_{k}||_2^2$$ 

   得到： $$\vec \mu_k^{<i+1>}=\frac {1}{|\mathbb I_k|}\sum_{j \in \mathbb I_k}\mathbf{\vec x}_j$$ ，其中 $$|\mathbb I_k|$$ 表示集合 $$|\mathbb I_k|$$ 的大小。

   这就是求平均值来更新簇中心。


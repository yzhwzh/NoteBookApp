# 线性可分支持向量机



1. 给定一个特征空间上的训练数据集 $$\mathbb D=\{(\mathbf {\vec x}_1,\tilde y_1),(\mathbf {\vec x}_2,\tilde y_2),\cdots,(\mathbf {\vec x}_N,\tilde y_N)\}$$ ，其中 $$\mathbf {\vec x}_i \in \mathcal X = \mathbb R^{n},\tilde y_i \in \mathcal Y=\{+1,-1\},i=1,2,\cdots,N$$ 。

    $$\mathbf {\vec x}_i$$ 为第  $$i$$ 个特征向量，也称作实例；  $$\tilde y_i$$ 为  $$\mathbf {\vec x}_i$$ 的类标记； $$(\mathbf {\vec x}_i,\tilde y_i)$$ 称作样本点。

   * 当 $$\tilde y_i=+1$$ 时，称 $$\mathbf {\vec x}_i$$ 为正例。
   * 当 $$\tilde y_i=-1$$ 时，称  $$\mathbf {\vec x}_i$$ 为负例。

   假设训练数据集是线性可分的，则学习的目标是在特征空间中找到一个分离超平面，能将实例分到不同的类。

   分离超平面对应于方程 $$\mathbf {\vec w} \cdot \mathbf {\vec x}+b=0$$ ， 它由法向量  $$\mathbf {\vec w}$$ 和截距  $$b$$ 决定，可以用 $$(\mathbf {\vec w},b)$$ 来表示。

2. 给定线性可分训练数据集，通过间隔最大化学习得到的分离超平面为： $$\mathbf {\vec w}^{*}\cdot \mathbf {\vec x}+b^{*}=0$$ ，

   相应的分类决策函数： $$f(\mathbf {\vec x})=\text{sign}(\mathbf {\vec w}^{*}\cdot \mathbf {\vec x}+b^{*})$$ ，称之为线性可分支持向量机。

3. 当训练数据集线性可分时，存在无穷个分离超平面可以将两类数据正确分开。
   * **感知机利用误分类最小的策略，求出分离超平面。但是此时的解有无穷多个。**
   * **线性可分支持向量机利用间隔最大化求得最优分离超平面，这样的解只有唯一的一个。**

#### 1.1 函数间隔

1. 可以将一个点距离分离超平面的远近来表示分类预测的可靠程度：
   * 一个点距离分离超平面越远，则该点的分类越可靠。
   * 一个点距离分离超平面越近，则该点的分类则不那么确信。
2. 在超平面 $$\mathbf {\vec w} \cdot \mathbf {\vec x}+b=0$$ 确定的情况下:
   * $$|\mathbf {\vec w} \cdot \mathbf {\vec x}_i+b|$$ 能够相对地表示点 $$\mathbf {\vec x}_i$$ 距离超平面的远近。
   *  $$\mathbf {\vec w} \cdot \mathbf {\vec x}_i+b$$ 的符号与类标记 $$\tilde y_i$$ 的符号是否一致能表示分类是否正确
     * $$\mathbf {\vec w} \cdot \mathbf {\vec x}_i+b  > 0$$ 时，即 $$\mathbf {\vec x}_i$$ 位于超平面上方，将 $$\mathbf {\vec x}_i$$ 预测为正类。

       此时若 $$\tilde y_i=+1$$ 则分类正确；否则分类错误。

     *  $$\mathbf {\vec w} \cdot \mathbf {\vec x}_i+b < 0$$ 时，即 $$\mathbf {\vec x}_i$$ 位于超平面下方，将 $$\mathbf {\vec x}_i$$ 预测为负类。

       此时若 $$\tilde y_i=-1$$ 则分类正确；否则分类错误。
3. 可以用 $$y(\mathbf {\vec w} \cdot \mathbf {\vec x}+b)$$ 来表示分类的正确性以及确信度，就是**函数间隔**的概念。
   * 符号决定了正确性。
   * 范数决定了确信度。
4. 对于给定的训练数据集 $$\mathbb D$$ 和超平面 $$(\mathbf {\vec w},b)$$ 
   * 定义超平面 $$(\mathbf {\vec w},b)$$ 关于样本点 $$(\mathbf {\vec x}_i,\tilde y_i)$$ 的函数间隔为 $$\hat\gamma_i=\tilde y_i(\mathbf {\vec w}\cdot \mathbf {\vec x}_i+b)$$ ： 
   * 定义超平面 $$(\mathbf {\vec w},b)$$ 关于训练集 $$\mathbb D$$ 的函数间隔为：超平面 $$(\mathbf {\vec w},b)$$ 关于 $$\mathbb D$$  中所有样本点 $$(\mathbf {\vec x}_i,\tilde y_i)$$  的函数间隔之最小值： $$\hat\gamma=\min_{\mathbb D} \hat\gamma_i$$  。

#### 1.2 几何间隔

1. 如果成比例的改变 $$\mathbf {\vec w}$$  和 $$b$$ ，比如将它们改变为 $$100 \mathbf {\vec w}$$ 和 $$100b$$ ，超平面 $$100\mathbf {\vec w}\cdot \mathbf {\vec x}+100b=0$$ 还是原来的超平面，但是函数间隔却成为原来的100倍。

   因此需要对分离超平面施加某些约束，如归一化，令 $$||\mathbf {\vec w}||_2=1$$ ，使得函数间隔是确定的。此时的函数间隔成为几何间隔。

2. 对于给定的训练数据集 $$\mathbb D$$ 和超平面 $$(\mathbf {\vec w},b)$$ 
   * 定义超平面 $$(\mathbf {\vec w},b)$$ 关于样本点 $$(\mathbf {\vec x}_i,\tilde y_i)$$ 的几何间隔为： $$\gamma_i=\tilde y_i(\frac{\mathbf {\vec w}}{||\mathbf {\vec w}||_2}\cdot \mathbf {\vec x}_i+\frac{b}{||\mathbf {\vec w}||_2})$$ 
   * 定义超平面 $$(\mathbf {\vec w},b)$$ 关于训练集 $$\mathbb D$$ 的几何间隔为：超平面 $$(\mathbf {\vec w},b)$$ 关于 $$\mathbb D$$ 中所有样本点 $$(\mathbf {\vec x}_i,\tilde y_i)$$ 的几何间隔之最小值： $$\gamma=\min_{\mathbb D} \gamma_i$$  。
3. 由定义可知函数间隔和几何间隔有下列的关系： $$\gamma_i=\frac{\hat\gamma_i}{||\mathbf {\vec w}||_2},\quad \gamma=\frac{\hat\gamma}{||\mathbf {\vec w}||_2}$$ 
   * 当 $$||\mathbf {\vec w}||_2=1$$ 时，函数间隔和几何间隔相等。
   * 当超平面参数 $$\mathbf {\vec w},b$$ 等比例改变时：
     * 超平面并没有变化。
     * 函数间隔也按比例改变。
     * 几何间隔保持不变。

#### 1.3 硬间隔最大化

1. 支持向量机学习基本思想：求解能够正确划分训练数据集并且几何间隔最大的分离超平面。几何间隔最大化又称作硬间隔最大化。

   对于线性可分的训练数据集而言，线性可分分离超平面有无穷多个（等价于感知机），但几何间隔最大的分离超平面是唯一的。

2. 几何间隔最大化的物理意义：**不仅将正负实例点分开，而且对于最难分辨的实例点（距离超平面最近的那些点），也有足够大的确信度来将它们分开。**
3. 求解几何间隔最大的分离超平面可以表示为约束的最优化问题： $$\max_{\mathbf {\vec w},b} \gamma\\ s.t. \quad \tilde y_i(\frac{\mathbf {\vec w}}{||\mathbf {\vec w}||_2}\cdot \mathbf {\vec x}_i+\frac{b}{||\mathbf {\vec w}||_2}) \ge \gamma, i=1,2,\cdots,N$$ 

   考虑几何间隔和函数间隔的关系，改写问题为： $$\max_{\mathbf {\vec w},b} \frac{\hat\gamma}{||\mathbf {\vec w}||_2}\\ s.t. \quad \tilde y_i(\mathbf {\vec w} \cdot \mathbf {\vec x}_i+b) \ge \hat\gamma, i=1,2,\cdots,N$$ 

4. 函数间隔 $$\hat \gamma$$ 的大小并不影响最优化问题的解。

   假设将 $$\mathbf {\vec w},b$$ 按比例的改变为 $$\lambda \mathbf {\vec w},\lambda b$$ ，此时函数间隔变成 $$\lambda \hat\gamma$$ （这是由于函数间隔的定义）：

   * 这一变化对求解最优化问题的不等式约束没有任何影响。
   * 这一变化对最优化目标函数也没有影响。

   因此取 $$\hat\gamma =1$$ ，则最优化问题改写为 $$\max_{\mathbf {\vec w},b} \frac{1}{||\mathbf {\vec w}||_2}\\ s.t. \quad \tilde y_i(\mathbf {\vec w} \cdot \mathbf {\vec x}_i+b) \ge 1, i=1,2,\cdots,N$$ ：

5. 注意到 $$\max \frac{1}{||\mathbf {\vec w}||_2}$$ 和  $$\min \frac 12 ||\mathbf {\vec w}||_2^{2}$$ 是等价的，于是最优化问题改写为： $$\min_{\mathbf {\vec w},b} \frac 12 ||\mathbf {\vec w}||_2^{2}\\ s.t. \quad \tilde y_i(\mathbf {\vec w} \cdot \mathbf {\vec x}_i+b) -1 \ge 0, i=1,2,\cdots,N$$ 

   这是一个凸二次规划问题。

6. 凸优化问题 ，指约束最优化问题： $$\min_{\mathbf {\vec w}}f(\mathbf {\vec w})\\  s.t. \quad g_j(\mathbf {\vec w}) \le0,j=1,2,\cdots,J\\  h_k(\mathbf {\vec w})=0,k=1,2,\cdots,K$$ 

   其中：

   * 目标函数 $$f(\mathbf {\vec w})$$ 和约束函数 $$g_j(\mathbf {\vec w})$$ 都是 $$\mathbb R^{n}$$ 上的连续可微的凸函数。
   * 约束函数 $$h_k(\mathbf {\vec w})$$ 是 $$\mathbb R^{n}$$ 上的仿射函数（仿射函数，即最高次数为1的多项式函数。常数项为零的仿射函数称为线性函数）。

     > $$h(\mathbf {\vec x})$$ 称为仿射函数，如果它满足 $$h(\mathbf {\vec x})=\mathbf {\vec a}\cdot\mathbf {\vec x}+b,\quad \mathbf {\vec a}\in \mathbb R^{n},b \in \mathbb R, x \in \mathbb R^{n}$$

   当目标函数 $$f(\mathbf {\vec w})$$ 是二次函数且约束函数 $$g_j(\mathbf {\vec w})$$ 是仿射函数时，上述凸最优化问题成为凸二次规划问题。

7. 线性可分支持向量机原始算法：
   * 输入：线性可分训练数据集 $$\mathbb D=\{(\mathbf {\vec x}_1,\tilde y_1),(\mathbf {\vec x}_2,\tilde y_2),\cdots,(\mathbf {\vec x}_N,\tilde y_N)\}$$ ，其中 $$\mathbf {\vec x}_i \in \mathcal X = \mathbb R^{n},\tilde y_i \in \mathcal Y=\{+1,-1\},i=1,2,\cdots,N$$ 
   * 输出：
     * 最大几何间隔的分离超平面
     * 分类决策函数
   * 算法步骤：
     * 构造并且求解约束最优化问题： $$\min_{\mathbf {\vec w},b} \frac 12 ||\mathbf {\vec w}||_2^{2}\\ s.t. \quad \tilde y_i(\mathbf {\vec w} \cdot \mathbf {\vec x}_i+b) -1 \ge 0, i=1,2,\cdots,N$$ 

       求得最优解 $$\mathbf {\vec w}^{*},b^{*}$$ 

     * 由此得到分离超平面： $$\mathbf {\vec w}^{*}\cdot \mathbf {\vec x}+b^{*}=0$$ ，以及分类决策函数 ： $$f(\mathbf {\vec x})=\text{sign}(\mathbf {\vec w}^{*}\cdot \mathbf {\vec x}+b^{*})$$ 。
8. 可以证明：若训练数据集 $$\mathbb D$$ 线性可分，则可将训练数据集中的样本点完全正确分开的最大间隔分离超平面存在且唯一。

> 感知机的最优化问题为误分类点到超平面的距离 $$-\frac{1}{||w||}y_i(w\cdot x_i+b)$$ ,若所有误分类点集合为M，那么所有误分类点到超平面S总距离为 $$-\frac{1}{||w||}\Sigma_{x_i\in M}y_i(w\cdot x_i+b) $$ 不考虑 $$\frac{1}{||w||}$$ ，就得到感知机的loss function. 对每一个点都进行梯度下降，找到正确分类平面

#### 1.4 支持向量

1. 在训练数据集线性可分的情况下，训练数据集的样本点中与分离超平面距离最近的样本点的实例称为支持向量。

   支持向量是使得约束条件等号成立的点，即 $$\tilde y_i(\mathbf {\vec w} \cdot \mathbf {\vec x}_i+b) -1=0$$ ：

   * 对于正实例点，支持向量位于超平面 $$H_1:\mathbf {\vec w} \cdot\mathbf {\vec x}+b=1$$ 
   * 对于负实例点，支持向量位于超平面 $$H_2:\mathbf {\vec w} \cdot\mathbf {\vec x}+b=-1$$ 

2. 超平面 $$H_1$$ 、 $$H_2$$ 称为间隔边界， 它们和分离超平面 $$\mathbf {\vec w} \cdot\mathbf {\vec x}+b=0$$ 平行，且没有任何实例点落在 $$H_1$$ 、 $$H_2$$ 之间。

   在 $$H_1$$ 、 $$H_2$$  之间形成一条长带，分离超平面位于长带的中央。长带的宽度称为 $$H_1$$ 、 $$H_2$$ 之间的距离，也即间隔，间隔大小为 $$\frac{2}{||\mathbf {\vec w}||}_2$$  。

   ![linear\_svm](http://www.huaxiaozhuan.com/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0/imgs/svm/linear_svm.png)

3. 在决定分离超平面时，只有支持向量起作用，其他的实例点并不起作用。
   * 如果移动支持向量，将改变所求的解。
   * 如果在间隔边界以外移动其他实例点，甚至去掉这些点，则解是不变的。
4. 由于支持向量在确定分离超平面中起着决定性作用，所以将这种分离模型称为支持向量机。
5. 支持向量的个数一般很少，所以支持向量机由很少的、重要的训练样本确定。

#### 1.5 对偶算法

1. 将线性可分支持向量机的最优化问题作为原始最优化问题，应用拉格朗日对偶性，通过求解对偶问题得到原始问题的最优解。这就是线性可分支持向量机的对偶算法。
2. 对偶算法的优点：
   * 对偶问题往往更容易求解。
   * 引入了核函数，进而推广到非线性分类问题。
3. 原始问题： $$\min_{\mathbf {\vec w},b} \frac 12 ||\mathbf {\vec w}||_2^{2}\\ s.t. \quad \tilde y_i(\mathbf {\vec w} \cdot \mathbf {\vec x}_i+b) -1 \ge 0, i=1,2,\cdots,N$$ 

   定义拉格朗日函数： $$L(\mathbf {\vec w},b,\vec\alpha)=\frac 12 ||\mathbf {\vec w}||_2^{2}-\sum_{i=1}^{N}\alpha_i\tilde y_i(\mathbf {\vec w} \cdot \mathbf {\vec x}_i +b)+\sum_{i=1}^{N} \alpha_i$$ 

   其中 $$\vec\alpha=(\alpha_1,\alpha_2,\cdots,\alpha_N)^{T}$$ 为拉格朗日乘子向量。

   * 根据拉格朗日对偶性，原始问题的对偶问题是极大极小问题： $$\max_\vec\alpha\min_{\mathbf {\vec w},b}L(\mathbf {\vec w},b,\vec\alpha)$$ 
   * 先求 $$\min_{\mathbf {\vec w},b}L(\mathbf {\vec w},b,\vec\alpha)$$ 。拉格朗日函数分别为 $$\mathbf {\vec w},b$$ 求偏导数，并令其等于 0, $$\nabla_{\mathbf {\vec w}}L(\mathbf {\vec w},b,\vec\alpha)=\mathbf {\vec w}-\sum_{i=1}^{N}\alpha_i\tilde y_i\mathbf {\vec x}_i =0\\ \nabla_bL(\mathbf {\vec w},b,\vec\alpha)=\sum_{i=1}^{N}\alpha_i\tilde y_i=0\\ \Rightarrow \mathbf {\vec w}=\sum_{i=1}^{N}\alpha_i\tilde y_i\mathbf {\vec x}_i,\quad \sum_{i=1}^{N}\alpha_i\tilde y_i=0$$ 
   * 代入拉格朗日函数： $$L(\mathbf {\vec w},b,\vec\alpha)=\frac 12 \sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_j\tilde y_i\tilde y_j(\mathbf {\vec x}_i \cdot \mathbf {\vec x}_j)-\sum_{i=1}^{N}\alpha_i\tilde y_i\left[\left(\sum_{j=1}^{N}\alpha_j\tilde y_j\mathbf {\vec x}_j\right)\cdot \mathbf {\vec x}_i+b\right]+\sum_{i=1}^{N} \alpha_i\\ =-\frac 12 \sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_j\tilde y_i\tilde y_j(\mathbf {\vec x}_i \cdot \mathbf {\vec x}_j) +\sum_{i=1}^{N} \alpha_i$$ 
   * 对偶问题极大值为： $$\max_{\vec\alpha}-\frac 12 \sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_j\tilde y_i\tilde y_j(\mathbf {\vec x}_i \cdot \mathbf {\vec x}_j) +\sum_{i=1}^{N} \alpha_i\\ s.t. \quad \sum_{i=1}^{N}\alpha_i\tilde y_i=0\\ \alpha_i \ge 0,i=1,2,\cdots,N$$ 

4. 设对偶最优化问题的 $$\vec\alpha$$  的解为 $$\vec\alpha^{*}=(\alpha_1^{*},\alpha_2^{*},\cdots,\alpha_N^{*})$$ ，则根据 `KKT` 条件有： $$\nabla_{\mathbf {\vec w}}L(\mathbf {\vec w}^*,b^*,\vec\alpha^*)=\mathbf {\vec w}^*-\sum_{i=1}^{N}\alpha_i^*\tilde y_i\mathbf {\vec x}_i =0\\ \nabla_bL(\mathbf {\vec w}^*,b^*,\vec\alpha^*)=\sum_{i=1}^{N}\alpha_i^*\tilde y_i=0\\ \alpha_i^*[\tilde y_i(\mathbf{\vec w}^*\cdot \mathbf{\vec x}_i+b^*)-1]=0,i=1,2,\cdots,N\\ \tilde y_i(\mathbf{\vec w}^*\cdot\mathbf{\vec x}_i+b^*)-1\ge 0,i=1,2,\cdots,N\\ \alpha_i^*\ge 0,i=1,2,\cdots,N$$ 
   * 根据第一个式子，有： $$\mathbf {\vec w}^{*}=\sum_{i=1}^{N}\alpha_i^{*}\tilde y_i\mathbf {\vec x}_i$$ 。
   * 由于 $$\vec\alpha^{*}$$ 不是零向量（若它为零向量，则  $$\mathbf {\vec w}^{*}$$ 也为零向量，矛盾），则必然存在某个 $$j$$ 使得 $$\alpha_j^{*} \gt 0$$ 。

     根据第三个式子，此时必有 $$\tilde y_j(\mathbf {\vec w}^{*} \cdot \mathbf {\vec x}_j+b^{*})-1=0$$ 。同时考虑到 $$\tilde y_j^{2}=1$$ ，得到 ： $$b^{*}=\tilde y_j-\sum_{i=1}^{N}\alpha_i^{*}\tilde y_i(\mathbf {\vec x}_i \cdot    \mathbf {\vec x}_j)$$ 

   * 于是分离超平面写作： $$\sum_{i=1}^{N}\vec\alpha_i^{*}\tilde y_i(\mathbf {\vec x} \cdot \mathbf {\vec x}_i)+b^{*}=0$$ 。

     分类决策函数写作： $$f(\mathbf {\vec x})=\text{sign}\left(\sum_{i=1}^{N}\alpha_i^{*}\tilde y_i(\mathbf {\vec x} \cdot \mathbf {\vec x}_i)+b^{*}\right)$$ 。

     上式称作线性可分支持向量机的对偶形式。

     可以看到：分类决策函数只依赖于输入 $$\mathbf {\vec x}$$ 和训练样本的内积。
5. 线性可分支持向量机对偶算法：
   * 输入：线性可分训练数据集 $$\mathbb D=\{(\mathbf {\vec x}_1,\tilde y_1),(\mathbf {\vec x}_2,\tilde y_2),\cdots,(\mathbf {\vec x}_N,\tilde y_N)\}$$ ，其中 $$\mathbf {\vec x}_i \in \mathcal X = \mathbb R^{n},\tilde y_i \in \mathcal Y=\{+1,-1\},i=1,2,\cdots,N$$ 
   * 输出：
     * 最大几何间隔的分离超平面
     * 分类决策函数
   * 算法步骤：
     * 构造并且求解约束最优化问题： $$\min_{\vec\alpha} \frac 12 \sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_j\tilde y_i\tilde y_j(\mathbf {\vec x}_i \cdot \mathbf {\vec x}_j) -\sum_{i=1}^{N} \alpha_i\\ s.t. \quad \sum_{i=1}^{N}\alpha_i\tilde y_i=0\\ \alpha_i \ge 0,i=1,2,\cdots,N$$ 

       求得最优解 $$\vec\alpha^{*}=(\alpha_1^{*},\alpha_2^{*},\cdots,\alpha_N^{*})^{T}$$ 。  

     * 计算  $$\mathbf {\vec w}^{*}=\sum_{i=1}^{N}\alpha_i^{*}\tilde y_i\mathbf {\vec x}_i$$ 。
     * 选择 $$\vec\alpha^{*}$$ 的一个正的分量 $$\alpha_j^{*} \gt 0$$ ，计算  $$b^{*}=\tilde y_j-\sum_{i=1}^{N}\alpha_i^{*}\tilde y_i(\mathbf {\vec x}_i \cdot \mathbf {\vec x}_j)$$ 。
     * 由此得到分离超平面： $$\mathbf {\vec w}^{*}\cdot \mathbf {\vec x}+b^{*}=0$$ ，以及分类决策函数 ： $$f(\mathbf {\vec x})=\text{sign}(\mathbf {\vec w}^{*}\cdot \mathbf {\vec x}+b^{*})$$ 。
6.  $$\mathbf {\vec w}^{*},b^{*} $$ 只依赖于 $$\alpha_i^{*} \gt 0$$ 对应的样本点 $$\mathbf {\vec x}_i,\tilde y_i$$ ，而其他的样本点对于 $$\mathbf {\vec w}^{*},b^{*} $$ 没有影响。
   * 将训练数据集里面对应于 $$\alpha_i^{*} \gt 0$$  的样本点对应的实例 $$\mathbf {\vec x}_i$$ 称为支持向量。
   * 对于 $$\alpha_i^{*} \gt 0$$ 的样本点，根据 $$\alpha_i^{*}[\tilde y_i(\mathbf {\vec w}^{*}\cdot \mathbf {\vec x}_i+b^{*})-1]=0$$ ，有： $$\mathbf {\vec w}^{*}\cdot \mathbf {\vec x}_i+b^{*}=\pm 1$$ 。

     即  $$\mathbf {\vec x}_i$$ 一定在间隔边界上。这与原始问题给出的支持向量的定义一致。


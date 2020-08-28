---
description: 对于线性不可分训练数据，线性支持向量机不再适用，但可以想办法将它扩展到线性不可分问题。（两个变量之间存在一次方函数关系，就称它们之间存在线性关系。）
---

# 线性支持向量机

#### 原始问题

1. 设训练集为 $$\mathbb D=\{(\mathbf {\vec x}_1,\tilde y_1),(\mathbf {\vec x}_2,\tilde y_2),\cdots,(\mathbf {\vec x}_N,\tilde y_N)\}$$ ，其中 $$\mathbf {\vec x}_i \in \mathcal X = \mathbb R^{n},\tilde y_i \in \mathcal Y=\{+1,-1\},i=1,2,\cdots,N$$ 。

   假设训练数据集不是线性可分的，这意味着某些样本点 $$(\mathbf {\vec x}_i,\tilde y_i)$$ 不满足函数间隔大于等于 1 的约束条件。

   * 对每个样本点 $$(\mathbf {\vec x}_i,\tilde y_i)$$ 引进一个松弛变量 $$\xi_i \ge 0$$ ，使得函数间隔加上松弛变量大于等于 1。

     即约束条件变成了： $$\tilde y_i(\mathbf {\vec w} \cdot \mathbf {\vec x}_i+b) \ge 1-\xi_i$$  。

   * 对每个松弛变量 $$\xi_i$$ ，支付一个代价 $$\xi_i$$ 。目标函数由原来的 $$\frac 12 ||\mathbf {\vec w}||^{2}_2$$ 变成： $$\min \frac 12 ||\mathbf {\vec w}||^{2}_2+C\sum_{i=1}^{N}\xi_i$$ 

     这里 $$C \gt 0$$ 称作惩罚参数，一般由应用问题决定。

     *  $$C $$ 值大时，对误分类的惩罚增大，此时误分类点凸显的更重要
     *  $$C $$ 值较大时，对误分类的惩罚增加，此时误分类点比较重要。
     *  $$C$$ 值较小时，对误分类的惩罚减小，此时误分类点相对不重要。

2. 相对于硬间隔最大化， $$\min \frac 12 ||\mathbf {\vec w}||^{2}_2+C\sum_{i=1}^{N}\xi_i$$ 称为软间隔最大化。

   于是线性不可分的线性支持向量机的学习问题变成了凸二次规划问题： $$\min_{\mathbf {\vec w},b,\vec\xi} \frac 12||\mathbf {\vec w}||^{2}_2+C\sum_{i=1}^{N}\xi_i\\ s.t. \quad \tilde y_i(\mathbf {\vec w}\cdot \mathbf {\vec x}_i+b) \ge 1-\xi_i,\quad i=1,2,\cdots,N\\ \xi_i \ge 0,\quad i=1,2,\cdots,N$$ 

   * 这称为线性支持向量机的原始问题。
   * 因为这是个凸二次规划问题，因此解存在。

     可以证明 $$\mathbf {\vec w}$$ 的解是唯一的； $$b$$ 的解不是唯一的， $$b$$ 的解存在于一个区间。

3. 对于给定的线性不可分的训练集数据，通过求解软间隔最大化问题得到的分离超平面为： $$\mathbf {\vec w}^{*}\cdot \mathbf {\vec x}+b^{*}=0$$ ，

   以及相应的分类决策函数： $$f(\mathbf {\vec x})=\mathbf {\vec w}^{*}\cdot \mathbf {\vec x}+b^{*}$$ ，称之为线性支持向量机。

* **线性支持向量机包含线性可分支持向量机**。
* 现实应用中训练数据集往往是线性不可分的，线性支持向量机具有更广泛的适用性。

#### 2.2 对偶问题

1. 定义拉格朗日函数为： $$L(\mathbf {\vec w},b,\vec\xi,\vec\alpha,\vec\mu)=\frac 12||\mathbf {\vec w}||_2^{2}+C\sum_{i=1}^{N}\xi_i-\sum_{i]1}^{N}\alpha_i[\tilde y_i(\mathbf {\vec w}_i \cdot \mathbf {\vec x}_i+b)-1+\xi_i]-\sum_{i=1}^{N}\mu_i\xi_i \\ \alpha_i \ge 0, \mu_i \ge 0$$ 

   原始问题是拉格朗日函数的极小极大问题；对偶问题是拉格朗日函数的极大极小问题。

   * 先求 $$L(\mathbf {\vec w},b,\vec\xi,\vec\alpha,\vec\mu)$$ 对 $$\mathbf {\vec w},b,\vec\xi$$ 的极小。根据偏导数为0： $$\nabla_{\mathbf {\vec w}}L(\mathbf {\vec w},b,\vec\xi,\vec\alpha,\vec\mu)=\mathbf {\vec w}-\sum_{i=1}^{N}\alpha_i\tilde y_i\mathbf {\vec x}_i=\mathbf{\vec 0}\\ \nabla_b L(\mathbf {\vec w},b,\vec\xi,\vec\alpha,\vec\mu)=-\sum_{i=1}^{N}\alpha_i\tilde y_i=0\\ \nabla_{\xi_i} L(\mathbf {\vec w},b,\vec\xi,\vec\alpha,\vec\mu)=C-\alpha_i-\mu_i=0$$ 

     得到：  $$\mathbf {\vec w}=\sum_{i=1}^{N}\alpha_i\tilde y_i\mathbf {\vec x}_i\\ \sum_{i=1}^{N}\alpha_i\tilde y_i=0\\ C-\alpha_i-\mu_i=0$$ 

   * 再求极大问题：将上面三个等式代入拉格朗日函数： $$\max_{\vec\alpha,\vec\mu}\min_{\mathbf {\vec w},b,\vec\xi}L(\mathbf {\vec w},b,\vec\xi,\vec\alpha,\vec\mu)=\max_{\vec\alpha,\vec\mu}\left[-\frac 12 \sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_j\tilde y_i\tilde y_j(\mathbf {\vec x}_i \cdot \mathbf {\vec x}_j)+\sum_{i=1}^{N}\alpha_i\right]$$ 

     于是得到对偶问题： $$\min_{\vec\alpha}\frac 12 \sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_j\tilde y_i\tilde y_j(\mathbf {\vec x}_i \cdot \mathbf {\vec x}_j)-\sum_{i=1}^{N}\alpha_i\\ s.t. \quad \sum_{i=1}^{N}\alpha_i\tilde y_i=0\\ 0 \le \alpha_i \le C, i=1,2,\cdots,N$$ 

2. 根据 `KKT` 条件： $$\nabla_{\mathbf {\vec w}}L(\mathbf {\vec w}^*,b^*,\vec\xi^*,\vec\alpha^*,\vec\mu^*)=\mathbf {\vec w}^*-\sum_{i=1}^{N}\alpha_i^*\tilde y_i\mathbf {\vec x}_i=\mathbf{\vec 0}\\ \nabla_b L(\mathbf {\vec w}^*,b^*,\vec\xi^*,\vec\alpha^*,\vec\mu^*)=-\sum_{i=1}^{N}\alpha_i^*\tilde y_i=0\\ \nabla_{\xi_i} L(\mathbf {\vec w}^*,b^*,\vec\xi^*,\vec\alpha^*,\vec\mu^*)=C-\alpha_i^*-\mu_i^*=0\\ \alpha_i^*[\tilde y_i(\mathbf{\vec w}^*\cdot \mathbf{\vec x}_i+b^*)-1+\xi_i^*]=0\\ \mu_i^*\xi_i^*=0\\ \tilde y_i(\mathbf{\vec w}^*\cdot\mathbf{\vec x}_i+b^*)-1+\xi_i^*\ge 0\\ \xi_i^*\ge 0\\ C\ge \alpha_i^*\ge 0\\ \mu_i^*\ge0\\ i=1,2,\cdots,N$$ 

   则有： $$\mathbf {\vec w}^{*}=\sum_{i=1}^{N}\vec\alpha^{*}\tilde y_i\mathbf {\vec x}_i$$ 。

   * 若存在 $$\vec\alpha^{*}$$ 的某个分量 $$\alpha_j^{*}, 0 \lt \alpha_j^{*} \lt C$$ ，则有： $$\mu_j^*=C-\alpha_j^*\gt 0$$ 。

     > 若 $$\vec\alpha^{*}$$ 的所有分量都等于 0 ，则得出 $$\mathbf {\vec w}^{*}$$ 为零，没有任何意义。
     >
     > 若  $$\vec\alpha^{*}$$ 的所有分量都等于 $$C$$ ，根据 $$\sum\alpha_i^*\tilde y_i=0$$ ，则要求 $$\sum \tilde y_i=0$$ 。这属于强加的约束，

     * 根据 $$\mu_j^*\xi_j^*=0$$ ，有 $$\xi_j^*=0$$ 。
     * 考虑 $$\alpha_j^*[\tilde y_j(\mathbf{\vec w}^*\cdot \mathbf{\vec x}_j+b^*)-1+\xi_j^*]=0$$ ，则有： $$b^{*}=\tilde y_j-\sum_{i=1}^{N}\vec\alpha^{*}\tilde y_i(\mathbf {\vec x}_i\cdot \mathbf {\vec x}_j)$$ 

   * 分离超平面为： $$\sum_{i=1}^{N}\vec\alpha^{*}\tilde y_i(\mathbf {\vec x}_i \cdot \mathbf {\vec x})+b^{*}=0$$  。
   * 分类决策函数为： $$f(\mathbf {\vec x})=\text{sign}\left[\sum_{i=1}^{N}\vec\alpha^{*}\tilde y_i(\mathbf {\vec x}_i \cdot \mathbf {\vec x})+b^{*}\right]$$ 。

3. 线性支持向量机对偶算法：
   * 输入：训练数据集 $$\mathbb D=\{(\mathbf {\vec x}_1,\tilde y_1),(\mathbf {\vec x}_2,\tilde y_2),\cdots,(\mathbf {\vec x}_N,\tilde y_N)\}$$ ，其中 $$\mathbf {\vec x}_i \in \mathcal X = \mathbb R^{n},\tilde y_i \in \mathcal Y=\{+1,-1\},i=1,2,\cdots,N$$ 
   * 输出：
     * 分离超平面
     * 分类决策函数
   * 算法步骤：
     * 选择惩罚参数  $$C\gt 0$$ ，构造并且求解约束最优化问题： $$\min_{\vec\alpha} \frac 12 \sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_j\tilde y_i\tilde y_j(\mathbf {\vec x}_i \cdot \mathbf {\vec x}_j) -\sum_{i=1}^{N} \alpha_i\\ s.t. \quad \sum_{i=1}^{N}\alpha_i\tilde y_i=0\\ C \ge \alpha_i \ge 0,i=1,2,\cdots,N$$ 

       求得最优解  $$\vec\alpha^{*}=(\alpha_1^{*},\alpha_2^{*},\cdots,\alpha_N^{*})^{T}$$ 。

     * 计算 ： $$\mathbf {\vec w}^{*}=\sum_{i=1}^{N}\alpha_i^{*}\tilde y_i\mathbf {\vec x}_i$$ 。
     * 选择 $$\vec\alpha^{*}$$ 的一个合适的分量 $$C \gt \alpha_j^{*} \gt 0$$ ，计算： $$b^{*}=\tilde y_j-\sum_{i=1}^{N}\alpha_i^{*}\tilde y_i(\mathbf {\vec x}_i \cdot \mathbf {\vec x}_j)$$  。

       > 可能存在多个符合条件的 。这是由于原始问题中，对  的解不唯一。所以
       >
       > 实际计算时可以取在所有符合条件的样本点上的平均值。

     * 由此得到分离超平面： $$\mathbf {\vec w}^{*}\cdot \mathbf {\vec x}+b^{*}=0$$ ，以及分类决策函数： $$f(\mathbf {\vec x})=\text{sign}(\mathbf {\vec w}^{*}\cdot \mathbf {\vec x}+b^{*})$$  。

#### 2.3 支持向量

1. 在线性不可分的情况下，对偶问题的解 $$\vec\alpha^{*}=(\alpha_1^{*},\alpha_2^{*},\cdots,\alpha_N^{*})^{T}$$ 中，对应于 $$\alpha_i^{*} \gt 0$$ 的样本点 $$(\mathbf {\vec x}_i,\tilde y_i)$$  的实例点 $$\mathbf {\vec x}_i$$ 称作支持向量，它是软间隔的支持向量。
2. 线性不可分的支持向量比线性可分时的情况复杂一些：

   根据 $$\nabla_{\xi_i} L(\mathbf {\vec w},b,\vec\xi,\vec\alpha,\vec\mu)=C-\alpha_i-\mu_i=0$$ ，以及 $$\mu_j^*\xi_j^*=0$$ ，则：

   * 若 $$\alpha_i^{*} \lt C$$ ，则 $$\mu_i \gt 0$$ ， 则松弛量 $$\xi_i =0$$ 。此时：支持向量恰好落在了间隔边界上。
   * 若 $$\alpha_i^{*} = C$$ ， 则 $$\mu_i =0$$ ，于是 $$\xi_i $$  可能为任何正数：
     * 若 $$0 \lt \xi_i \lt 1$$ ，则支持向量落在间隔边界与分离超平面之间，分类正确。
     * 若 $$\xi_i =1$$ ，则支持向量落在分离超平面上。
     * 若 $$\xi_i > 1$$ ，则支持向量落在分离超平面误分类一侧，分类错误。

#### 2.4 合页损失函数

1. 定义取正函数为： $$\text{plus}(z)= \begin{cases} z, & z \gt 0 \\ 0, & z \le 0 \end{cases}$$ 

   定义合页损失函数为： $$L(\tilde y,\hat y)=\text{plus}(1-\tilde y \hat y)$$  ，其中 $$\tilde y$$ 为样本的标签值， $$\hat y$$ 为样本的模型预测值。

   则线性支持向量机就是最小化目标函数： $$\sum_{i=1}^{N}\text{plus}(1-\tilde y_i(\mathbf {\vec w}\cdot\mathbf {\vec x}_i+b))+\lambda||\mathbf {\vec w}||^{2}_2,\quad \lambda \gt 0$$ 

2. 合页损失函数的物理意义：
   * 当样本点 $$(\mathbf {\vec x}_i,\tilde y_i)$$ 被正确分类且函数间隔（确信度） $$\tilde y_i(\mathbf {\vec w}\cdot\mathbf {\vec x}_i+b)$$  大于 1 时，损失为0
   * 当样本点 $$(\mathbf {\vec x}_i,\tilde y_i)$$ 被正确分类且函数间隔（确信度） $$\tilde y_i(\mathbf {\vec w}\cdot\mathbf {\vec x}_i+b)$$ 小于等于 1 时损失为 $$1-\tilde y_i(\mathbf {\vec w}\cdot\mathbf {\vec x}_i+b)$$ 
   * 当样本点 $$(\mathbf {\vec x}_i,\tilde y_i)$$ 未被正确分类时损失为 $$1-\tilde y_i(\mathbf {\vec w}\cdot\mathbf {\vec x}_i+b)$$ 
3. 可以证明：线性支持向量机原始最优化问题等价于最优化问题： $$\min_{\mathbf {\vec w},b}\sum_{i=1}^{N}\text{plus}(1-\tilde y_i(\mathbf {\vec w}\cdot\mathbf {\vec x}_i+b))+\lambda||\mathbf {\vec w}||^{2}_2,\quad \lambda \gt 0$$ 
4. 合页损失函数图形如下：

   * 感知机的损失函数为 $$\text{plus}(-\tilde y(\mathbf {\vec w}\cdot\mathbf {\vec x}+b))$$ ，相比之下合页损失函数不仅要分类正确，而且要确信度足够高（确信度为1）时，损失才是0。即合页损失函数对学习有更高的要求。
   * 0-1损失函数通常是二分类问题的真正的损失函数，合页损失函数是0-1损失函数的上界。
     * 因为0-1损失函数不是连续可导的，因此直接应用于优化问题中比较困难。
     * 通常都是用0-1损失函数的上界函数构成目标函数，这时的上界损失函数又称为代理损失函数。

5. 理论上`SVM` 的目标函数可以使用梯度下降法来训练。但存在三个问题：
   * 合页损失函数部分不可导。这可以通过`sub-gradient descent` 来解决。
   * 收敛速度非常慢。
   * 无法得出支持向量和非支持向量的区别。

![hinge\_loss](http://www.huaxiaozhuan.com/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0/imgs/svm/hinge_loss.png)


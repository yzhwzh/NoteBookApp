# 非线性支持向量机

1. 非线性分类问题是指利用非线性模型才能很好的进行分类的问题。
2. 对于给定的训练集  ，其中 ，如果能用  中的一个超曲面将正负实例正确分开，则称这个问题为非线性可分问题。
3. 设原空间为 ，新的空间为  。定义

   从原空间到新空间的变换（映射）为： 。

   则经过变换 ：

   * 原空间  变换为新空间  ， 原空间中的点相应地变换为新空间中的点。
   * 原空间中的椭圆  变换为新空间中的直线  。
   * 若在变换后的新空间，直线  可以将变换后的正负实例点正确分开，则原空间的非线性可分问题就变成了新空间的线性可分问题。

4. 用线性分类方法求解非线性分类问题分两步：

   * 首先用一个变换将原空间的数据映射到新空间。
   * 再在新空间里用线性分类学习方法从训练数据中学习分类模型。

   这一策略称作核技巧。

## 核函数

### **核函数定义**

1. 设 $$\mathcal X$$ 是输入空间（欧氏空间 $$\mathbb R^{n}$$ 的子集或者离散集合）， $$\mathcal H$$  为特征空间（希尔伯特空间）。若果存在一个从 $$\mathcal X$$ 到 $$\mathcal H$$ 的映射  $$\phi(\mathbf {\vec x}):\mathcal X \rightarrow \mathcal H$$ ，使得所有的 $$\mathbf { \vec x} ,\mathbf{\vec z} \in\mathcal X$$ ， 函数 $$K(\mathbf {\vec x},\mathbf {\vec z})=\phi(\mathbf {\vec x})\cdot \phi(\mathbf {\vec z})$$ ，则称 $$K(\mathbf {\vec x},\mathbf {\vec z})$$ 为核函数。

   即：核函数将原空间中的任意两个向量 $$\mathbf { \vec x} ,\mathbf{\vec z}$$ ，映射为特征空间中对应的向量之间的内积。

2. 实际任务中，通常直接给定核函数  $$K(\mathbf {\vec x},\mathbf {\vec z})$$ ，然后用解线性分类问题的方法求解非线性分类问题的支持向量机。
   * 学习是隐式地在特征空间进行的，不需要显式的定义特征空间和映射函数。
   * 通常直接计算 $$K(\mathbf {\vec x},\mathbf {\vec z})$$ 比较容易，反而是通过 $$ \phi(\mathbf {\vec x})$$ 和 $$\phi(\mathbf {\vec z})$$ 来计算 $$K(\mathbf {\vec x},\mathbf {\vec z})$$ 比较困难。
     * 首先特征空间 $$\mathcal H$$ 一般是高维的，甚至是无穷维的，映射 $$\phi(\mathbf {\vec x})$$ 不容易定义。
     * 其次核函数关心的是希尔伯特空间两个向量的内积，而不关心这两个向量的具体形式。因此对于给定的核函数，特征空间 $$\mathcal H$$ 和 映射函数 $$\phi(\mathbf {\vec x})$$ 取法并不唯一。
       * 可以取不同的特征空间 $$\mathcal H$$ 。
       * 即使是在同一个特征空间 $$\mathcal H$$ 里，映射函数 $$\phi(\mathbf {\vec x})$$ 也可以不同。
3. 在线性支持向量机的对偶形式中，无论是目标函数还是决策函数都只涉及输入实例之间的内积。
   * 在对偶问题的目标函数中的内积 $$\mathbf {\vec x}_i \cdot \mathbf {\vec x}_j$$ 可以用核函数 $$K(\mathbf {\vec x}_i,\mathbf {\vec x}_j)=\phi(\mathbf {\vec x}_i)\cdot \phi(\mathbf {\vec x}_j)$$ 来代替。

     此时对偶问题的目标函数成为： $$L(\vec\alpha)=\frac 12 \sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_j\tilde y_i\tilde y_jK(\mathbf {\vec x}_i,\mathbf {\vec x}_j)-\sum_{i=1}^{N}\alpha_i$$ 

   * 分类决策函数中的内积也可以用核函数代替： $$f(\mathbf {\vec x})=\text{sign}\left(\sum_{i=1}^{N}\alpha_i^{*}\tilde y_iK(\mathbf {\vec x}_i,\mathbf {\vec x}_j)+b^{*}\right)$$ 。
4. 核函数替代法，等价于：
   * 首先经过映射函数 $$\phi$$ 将原来的输入空间变换到一个新的特征空间。
   * 然后将输入空间中的内积 $$\mathbf {\vec x}_i \cdot \mathbf {\vec x}_j$$ 变换为特征空间中的内积 $$\phi(\mathbf {\vec x}_i) \cdot \phi(\mathbf {\vec x}_j) $$ 。
   * 最后在新的特征空间里从训练样本中学习线性支持向量机。
5. 若映射函数 $$\phi$$ 为非线性函数，则学习到的含有核函数的支持向量机是非线性分类模型。

   若映射函数  $$\phi$$ 为线性函数，则学习到的含有核函数的支持向量机依旧是线性分类模型。

**3.1.2 核函数选择**

1. 在实际应用中，核函数的选取往往依赖领域知识，最后通过实验验证来验证核函数的有效性。
2. 若已知映射函数 $$\phi$$ ，那么可以通过 $$\phi(\mathbf {\vec x})$$ 和 $$\phi(\mathbf {\vec z})$$ 的内积求得核函数 $$K(\mathbf {\vec x},\mathbf {\vec z})$$ 。现在问题是：不用构造映射 $$\phi$$ ， 那么给定一个函数 $$K(\mathbf {\vec x},\mathbf {\vec z})$$ 判断它是否是一个核函数？

   即：  $$K(\mathbf {\vec x},\mathbf {\vec z})$$ 满足什么条件才能成为一个核函数？

   可以证明： 设 $$K:\mathcal X \times \mathcal X \rightarrow \mathbb R$$ 是对称函数， 则 $$K(\mathbf {\vec x},\mathbf {\vec z})$$ 为正定核函数的充要条件是：对任意 $$\mathbf {\vec x}_i \in \mathcal X,i=1,2,\cdots,N$$ ，  $$K(\mathbf {\vec x},\mathbf {\vec z})$$ 对应的 `Gram` 矩阵： $$K=[K(\mathbf {\vec x}_i,\mathbf {\vec x}_j)]_{N\times N}$$ 是半正定矩阵。

3. 对于一个具体函数  $$K(\mathbf {\vec x},\mathbf {\vec z})$$ 来说，检验它为正定核函数并不容易。因为要求对任意有限输入集 $$\{\mathbf {\vec x}_1,\mathbf {\vec x}_2,\cdots,\mathbf {\vec x}_N\}$$ 来验证 $$ K(\cdot,\cdot)$$ 对应的 `Gram` 矩阵是否为半正定的。

   因此，实际问题中往往应用已有的核函数。

4. 常用核函数：
   * 多项式核函数： $$K(\mathbf {\vec x},\mathbf {\vec z})=(\mathbf {\vec x}\cdot \mathbf {\vec z}+1)^{p}$$ 。

     对应的支持向量机是一个  次多项式分类器。

   * 高斯核函数： $$K(\mathbf {\vec x},\mathbf {\vec z})=\exp(-\frac{||\mathbf{\vec x}-\mathbf{\vec z}||^{2}}{2\sigma^{2}})$$ 
     * 它是最常用的核函数，对应于无穷维空间中的点积。
     * 它也被称作径向基函数`radial basis function:RBF` ，因为其值从 $$\mathbf{\vec x}$$ 沿着 $$\mathbf{\vec z}$$ 向外辐射的方向减小。
     * 对应的支持向量机是高斯径向基函数分类器\(`radial basis function`\) 。
   * `sigmod`核函数： $$K(\mathbf {\vec x},\mathbf {\vec z})=\tanh(\gamma(\mathbf  {\vec x}\cdot \mathbf {\vec z})+r)$$ 。

     对应的支持向量机实现的就是一种神经网络。

#### 3.2 学习算法

1. 非线性支持向量机学习算法：
   * 输入：训练数据集 $$\mathbb D=\{(\mathbf {\vec x}_1,\tilde y_1),(\mathbf {\vec x}_2,\tilde y_2),\cdots,(\mathbf {\vec x}_N,\tilde y_N)\}$$ ，其中  $$\mathbf {\vec x}_i \in \mathcal X = \mathbb R^{n},\tilde y_i \in \mathcal Y=\{+1,-1\},i=1,2,\cdots,N$$ 。
   * 输出：分类决策函数
   * 算法步骤：

     * 选择适当的核函数 $$K(\mathbf { \vec x} ,\mathbf{\vec z})$$ 和惩罚参数 $$C\gt 0$$ ，构造并且求解约束最优化问题： $$\min_{\vec\alpha} \frac 12 \sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_j\tilde y_i\tilde y_jK(\mathbf {\vec x}_i,\mathbf{\vec x}_j) -\sum_{i=1}^{N} \alpha_i\\ s.t. \quad \sum_{i=1}^{N}\alpha_i\tilde y_i=0\\ C \ge \alpha_i \ge 0,i=1,2,\cdots,N$$ 

     求得最优解 $$\vec\alpha^{*}=(\alpha_1^{*},\alpha_2^{*},\cdots,\alpha_N^{*})^{T}$$ 

     > 当 $$K(\mathbf { \vec x} ,\mathbf{\vec z})$$ 是正定核函数时，该问题为凸二次规划问题，解是存在的。

     * 计算： $$\mathbf {\vec w}^{*}=\sum_{i=1}^{N}\alpha_i^{*}\tilde y_i\mathbf {\vec x}_i$$  。
     * 选择 $$\vec\alpha^{*}$$ 的一个合适的分量 $$C \gt \alpha_j^{*} \gt 0$$ ，计算： $$b^{*}=\tilde y_j-\sum_{i=1}^{N}\alpha_i^{*}\tilde y_iK(\mathbf {\vec x}_i,\mathbf{\vec x}_j)$$ 。
     * 构造分类决策函数 ： $$f(\mathbf {\vec x})=\text{sign}\left(\sum_{i=1}^{N}\alpha_i^{*} \tilde y_iK(\mathbf {\vec x}_i,\mathbf{\vec x})+b^{*}\right)$$ 。




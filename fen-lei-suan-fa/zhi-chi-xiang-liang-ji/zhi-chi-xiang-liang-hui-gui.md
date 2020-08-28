# 支持向量回归

1. 支持向量机不仅可以用于分类问题，也可以用于回归问题。
2. 给定训练数据集 $$\mathbb D=\{(\mathbf {\vec x}_1,\tilde y_1),(\mathbf {\vec x}_2,\tilde y_2),\cdots,(\mathbf {\vec x}_N,\tilde y_N)\}$$  ，其中 $$\mathbf {\vec x}_i  \in \mathcal X = \mathbb R^{n},\tilde y_i \in \mathcal Y=\mathbb R $$ 。
   * 对于样本 $$(\mathbf{\vec x}_i,\tilde y_i)$$ ，传统的回归模型通常基于模型输出 $$f(\mathbf{\vec x}_i)$$ 与真实输出 $$\tilde y_i$$ 之间的差别来计算损失。当且仅当 $$f(\mathbf{\vec x}_i)$$  与  $$\tilde y_i$$ 完全相同时，损失才为零。
   * 支持向量回归\(`Support Vector Regression:SVR`\)不同：它假设能容忍 $$f(\mathbf{\vec x}_i)$$ 与 $$\tilde y_i$$ 之间最多有 $$\epsilon$$ 的偏差。仅当 $$|f(\mathbf{\vec x}_i)-\tilde y_i| \gt \epsilon$$ 时，才计算损失。

     支持向量回归相当于以 $$f(\mathbf{\vec x}_i)$$ 为中心，构建了一个宽度为 $$2\epsilon$$ 的间隔带。若训练样本落在此间隔带内则被认为是预测正确的。

## 原始问题

1. `SVR`问题形式化为： $$f(\mathbf{\vec x})=\mathbf{\vec w}\cdot \mathbf{\vec x}+b\\ \min_{\mathbf{\vec w},b}\frac 12 ||\mathbf{\vec w}||_2^{2}+C\sum_{i=1}^{N}L_\epsilon\left(f(\mathbf{\vec x}_i)-\tilde y_i\right)$$ 

   其中：

   *  $$C$$ 为罚项常数。
     * 若 $$C$$ 较大，则倾向于 $$f(\mathbf{\vec x}_i)$$ 与 $$\tilde y_i$$ 之间较小的偏差
     * 若 $$C$$ 较小，则能容忍 $$f(\mathbf{\vec x}_i)$$ 与 $$\tilde y_i$$ 之间较大的偏差
   * $$L_\epsilon$$ 为损失函数。其定义为： $$L_\epsilon(z)=\begin{cases} 0&, \text{if} |z| \le \epsilon\\ |z|-\epsilon&,\text{else} \end{cases}$$ 

     > 线性回归中，损失函数为 $$L(z)=z^{2}$$

     ![L\_epsilon](http://www.huaxiaozhuan.com/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0/imgs/svm/L_epsilon.png)

2. 引入松弛变量 ，将上式写做： $$\min_{\mathbf{ \vec w},b,\xi_i,\hat\xi_i}\frac 12 ||\mathbf{\vec w}||_2^{2}+C\sum_{i=1}^{N} (\xi_i+\hat \xi_i)\\ s.t. f(\mathbf{\vec x}_i)-\tilde y_i \le \epsilon+\xi_i,\\ \tilde y_i-f(\mathbf{\vec x}_i) \le \epsilon+\hat\xi_i,\\ \xi_i \ge 0,\hat\xi_i \ge 0, i=1,2,\cdots,N$$ 

   这就是 `SVR`原始问题。

#### 4.2 对偶问题

1. 引入拉格朗日乘子 $$\mu_i \ge 0,\hat \mu_i \ge 0,\alpha_i \ge 0,\hat\alpha_i \ge 0$$ ，定义拉格朗日函数： $$L(\mathbf{\vec w},b,\vec\alpha,\hat{\vec\alpha},\vec\xi,\hat{\vec\xi},\vec\mu,\hat{\vec\mu}) =\frac 12 ||\mathbf{\vec w}||_2^{2}+C\sum_{i=1}^{N}( \xi_i+\hat\xi_i)-\sum_{i=1}^{N}\mu_i\xi_i-\sum_{i-1}^{N}\hat\mu_i\hat\xi_i\\ +\sum_{i=1}^{N}\alpha_i\left( f(\mathbf{\vec x}_i)-\tilde y_i-\epsilon-\xi_i \right)+\sum_{i-1}^{N}\hat\alpha_i\left(\tilde y_i-f(\mathbf{\vec x}_i)-\epsilon-\hat\xi_i\right)$$ 

   根据拉格朗日对偶性，原始问题的对偶问题是极大极小问题： $$\max_{\vec\alpha,\hat{\vec\alpha}}\min_{\mathbf {\vec w},b,\vec\xi,\hat{\vec\xi}}L(\mathbf{\vec w},b,\vec\alpha,\hat{\vec\alpha},\vec\xi,\hat{\vec\xi},\vec\mu,\hat{\vec\mu})$$ 

2. 先求极小问题：根据 $$L(\mathbf{\vec w},b,\vec\alpha,\hat{\vec\alpha},\vec\xi,\hat{\vec\xi},\vec\mu,\hat{\vec\mu})$$ 对 $$\mathbf {\vec w},b,\vec\xi,\hat{\vec\xi}$$ 偏导数为零可得： $$\mathbf {\vec w}=\sum_{i=1}^{N}(\hat \alpha_i-\alpha_i)\mathbf{\vec x}_i\\ 0=\sum_{i=1}^{N}(\hat \alpha_i-\alpha_i)\\ C=\alpha_i+\mu_i\\ C=\hat\alpha_i+\hat\mu_i$$ 
3. 再求极大问题（取负号变极小问题）： $$\min_{\vec\alpha,\hat{\vec\alpha}} \sum_{i=1}^{N}\left[\tilde y_i(\hat\alpha_i-\alpha_i)-\epsilon(\hat\alpha_i+\alpha_i)\right]-\frac 12 \sum_{i=1}^{N}\sum_{j=1}^{N}(\hat\alpha_i-\alpha_i)(\hat\alpha_j-\alpha_j)\mathbf{\vec x}_i^{T}\mathbf{\vec x}_j\\ s.t. \sum_{i=1}^{N}(\hat\alpha_i-\alpha_i)=0\\ 0 \le \alpha_i,\hat\alpha_i \le C$$ 
4. 上述过程需要满足`KKT`条件，即： $$\begin{cases} \alpha_i\left( f(\mathbf{\vec x}_i)-\tilde y_i-\epsilon-\xi_i \right)=0\\ \hat\alpha_i\left(\tilde y_i-f(\mathbf{\vec x}_i)-\epsilon-\hat\xi_i\right)=0\\ \alpha_i\hat\alpha_i=0\\ \xi_i\hat\xi_i=0\\ (C-\alpha_i)\xi_i=0\\ (C-\hat\alpha_i)\hat\xi_i=0 \end{cases}$$ 
5. 可以看出：
   * 当样本  $$(\mathbf{\vec x}_i,\tilde y_i)$$ 不落入 $$\epsilon$$ 间隔带中时，对应的 $$\alpha_i,\hat\alpha_i$$ 才能取非零值：
     * 当且仅当 $$f(\mathbf{\vec x}_i)-\tilde y_i-\epsilon-\xi_i=0$$ 时， $$\alpha_i$$ 能取非零值
     * 当且仅当 $$\tilde y_i-f(\mathbf{\vec x}_i)-\epsilon-\hat\xi_i=0$$ 时， $$\hat\alpha_i$$ 能取非零值
   * 此外约束  $$f(\mathbf{\vec x}_i)-\tilde y_i-\epsilon-\xi_i=0$$ 与  $$\tilde y_i-f(\mathbf{\vec x}_i)-\epsilon-\hat\xi_i=0$$ 不能同时成立，因此 $$\alpha_i,\hat\alpha_i$$ 中至少一个为零。
6. 设最终解 $$\vec\alpha$$ 中，存在 $$C \gt \alpha_j \gt 0$$ ，则有： $$b=\tilde y_j+\epsilon-\sum_{i=1}^{N}(\hat\alpha_i-\alpha_j)\mathbf{\vec x}_i^{T}\mathbf{\vec x}_j\\ f(\mathbf {\vec x})=\sum_{i=1}^{N}(\hat\alpha_i-\alpha_i)\mathbf{\vec x}_i^{T}\mathbf{\vec x}+b$$ 
7. 最后若考虑使用核技巧，则`SVR`可以表示为： $$f(\mathbf {\vec x})=\sum_{i=1}^{N}(\hat\alpha_i-\alpha_i)K(\mathbf{\vec x}_i,\mathbf{\vec x})+b$$ 。




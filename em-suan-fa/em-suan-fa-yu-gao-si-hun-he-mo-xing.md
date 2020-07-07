# EM算法与高斯混合模型

## 高斯混合模型

 高斯混合模型\(`Gaussian mixture model,GMM`\)：指的是具有下列形式的概率分布模型：

$$
P(y;\theta)=\sum_{k=1}^{K}\alpha_k\phi(y;\theta_k)
$$

其中 $$\alpha_k$$ 是系数，满足 ：

* $$\alpha_k \ge 0,\sum_{k=1}^K \alpha_k=1$$ 。
*  $$\phi(y;\theta_k)$$ 是高斯分布密度函数，称作第 $$k$$ 个分模型， $$\theta_k=(\mu_k,\sigma_k^{2})$$ ：

$$
\phi(y;\theta_k)=\frac{1}{\sqrt{2\pi}\sigma_k}\exp\left(-\frac{(y-\mu_k)^{2}}{2\sigma_k^{2}}\right)
$$

如果用其他的概率分布密度函数代替上式中的高斯分布密度函数，则称为一般混合模型。

## 参数估计

1. 假设观察数据 $$\mathbb Y=\{y_1,y_2,\cdots,y_N\}$$ 由高斯混合模型 $$P(y;\theta)=\sum_{k=1}^{K}\alpha_k\phi(y;\theta_k)$$ 生成，其中 $$\theta=(\alpha_1,\alpha_2,\cdots,\alpha_K;\theta_1,\theta_2,\cdots,\theta_K)$$ 。

   可以通过`EM`算法估计高斯混合模型的参数 $$\theta$$ 。

2. 可以设想观察数据 $$y_i$$ 是这样产生的：

   * 首先以概率 $$\alpha_k$$ 选择第 $$k$$ 个分模型  $$\phi(y;\theta_k)$$ 。
   * 然后以第 $$k$$ 个分模型的概率分布 $$\phi(y;\theta_k)$$ 生成观察数据 $$y_i$$ 。

   这样，观察数据 $$y_i$$ 是已知的，观测数据 $$y_i$$ 来自哪个分模型是未知的。

   对观察变量 $$y$$ ，定义隐变量  $$z$$ ，其中  $$p(z=k)=\alpha_k$$ 。

3. 完全数据的对数似然函数为： $$P(y=y_j,z=k;\theta)=\alpha_k\frac{1}{\sqrt{2\pi}\sigma_k}\exp\left(-\frac{(y_j-\mu_k)^{2}}{2\sigma_k^{2}}\right)$$ 

   其对数为： $$\log P(y=y_j,z=k;\theta)=\log \alpha_k-\log\sqrt{2\pi}\sigma_k -\frac{(y_j-\mu_k)^{2}}{2\sigma_k^{2}}$$ 

   后验概率为： $$P(z=k\mid y=y_j;\theta^{<i>})=\frac{\alpha_k\frac{1}{\sqrt{2\pi}\sigma_k^{<i>}}\exp\left(-\frac{(y_j-\mu_k^{<i>})^{2}}{2\sigma_k^{^{<i>}2}}\right)}{\sum_{t=1}^K\alpha_t\frac{1}{\sqrt{2\pi}\sigma_t^{<i>}}\exp\left(-\frac{(y_j-\mu_t^{<i>})^{2}}{2\sigma_t^{^{<i>}2}}\right)}$$ 

   即： $$P(z=k\mid y=y_j;\theta^{<i>})=\frac{P(y=y_j,z=k;\theta^{<t>})}{\sum_{t=1}^KP(y=y_j,z=t;\theta^{})}$$ 。

   则 $$Q$$ 函数为： $$Q(\theta,\theta^{<i>})=\sum_{j=1}^N\left(\sum_z P(z\mid y=y_j;\theta^{<i>})\log P(y=y_j,z;\theta) \right)\\ =\sum_{j=1}^N\sum_{k=1}^K P(z=k\mid y=y_j;\theta^{<i>})\left(\log \alpha_k-\log\sqrt{2\pi}\sigma_k -\frac{(y_j-\mu_k)^{2}}{2\sigma_k^{2}}\right)$$ 

   求极大值： $$\theta^{<i+1>}=\arg\max_{\theta}Q(\theta,\theta^{<i>})$$ 。

   根据偏导数为 0，以及 $$\sum_{k=1}^{K}\alpha_k=1$$ 得到：

   * $$\alpha_k$$ ： $$\alpha_k^{<i+1>}=\frac{n_k}{N}$$ 

     其中： $$n_k=\sum_{j=1}^NP(z=k\mid y=y_j;\theta^{<i>})$$  ，其物理意义为：所有的观测数据  $$Y$$ 中，产生自第 $$k$$ 个分模型的观测数据的数量。

   * $$\mu_k$$ ： $$\mu_k^{<i+1>}=\frac{\overline {Sum}_k}{n_k}$$ 

     其中： $$\overline {Sum}_k=\sum_{j=1}^N y_j P(z=k\mid y=y_j;\theta^{<i>})$$ ，其物理意义为：所有的观测数据  $$Y$$ 中，产生自第 $$k$$ 个分模型的观测数据的总和。

   * $$\sigma^2$$ ： $$\sigma_k^{<i+1>2}=\frac{\overline {Var}_k}{n_k}$$ 

     其中： $$\overline {Var}_k=\sum_{j=1}^N (y_j-\mu_k^{<i>})^2P(z=k\mid y=y_i;\theta^{<i>})$$ ，其物理意义为：所有的观测数据 $$Y$$ 中，产生自第 $$k$$ 个分模型的观测数据，偏离第 $$k$$ 个模型的均值（ $$\mu_k^{<i>}$$ ）的平方和。

4. 高斯混合模型参数估计的`EM`算法：
   * 输入：
     * 观察数据 $$\mathbb Y=\{y_1,y_2,\cdots,y_N\}$$ 
     * 高斯混合模型的分量数 $$K$$ 
   * 输出：高斯混合模型参数 $$\theta=(\alpha_1,\alpha_2,\cdots,\alpha_K;\mu_1,\mu_2,\cdots,\mu_K;\sigma^2_1,\sigma^2_2,\cdots,\sigma^2_K)$$ 
   * 算法步骤：
     * 随机初始化参数  $$\theta^{<0>}$$ 。
     * 根据 $$\theta^{<i>}$$ 迭代求解  $$\theta^{<i+1>}$$ ，停止条件为：对数似然函数值或者参数估计值收敛。
     * $$\alpha_k^{<i+1>}=\frac{n_k}{N},\;\mu_k^{<i+1>}=\frac{\overline {Sum}_k}{n_k},\;\sigma_k^{<i+1>2}=\frac{\overline {Var}_k}{n_k}$$ 

       其中：

       * $$n_k=\sum_{j=1}^NP(z=k\mid y=y_j;\theta^{<i>})$$ 。

         其物理意义为：所有的观测数据 $$Y$$ 中，产生自第 $$k$$ 个分模型的观测数据的数量。

       * $$\overline {Sum}_k=\sum_{j=1}^N y_j P(z=k\mid y=y_j;\theta^{<i>})$$ 。

         其物理意义为：所有的观测数据 $$Y$$ 中，产生自第 $$k$$ 个分模型的观测数据的总和。

       *  $$\overline {Var}_k=\sum_{j=1}^N (y_j-\mu_k^{<i>})^2P(z=k\mid y=y_i;\theta^{<i>})$$ 。

         其物理意义为：所有的观测数据 $$Y$$ 中，产生自第 $$k$$ 个分模型的观测数据，偏离第 $$k$$ 个模型的均值（ $$\mu_k^{<i>}$$ ）的平方和。

